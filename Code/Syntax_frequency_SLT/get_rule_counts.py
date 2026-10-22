"""Extract lexical tokens, dependency-based syntax rules, and sentence metrics.

Input:  Data/parses/<SubjectID>.txt produced by the Stanford lexicalized parser.
Output: Data/syntaxrules.csv, Data/wordsrules_{content,nocontent}.csv,
        Data/alternative_metrics.csv, and Data/pos_entropy.csv.

IMPORTANT: This script extracts rule occurrences; it does NOT compute the
Switchboard-based syntax-frequency lookup values. Those are merged in SLT.R.
The parser's original output format and grouping are assumed throughout.
"""

from nltk.stem import WordNetLemmatizer
import nltk
import pandas as pd
import os
from collections import Counter, defaultdict
import numpy as np


def is_switchboard(person):
    """Identify subject IDs assigned to Switchboard reference transcripts."""
    return person.startswith("B") or person.startswith("switch") or person.startswith("Switch") or person.startswith("ss")

def is_content_word(word, pos, deps):
    """Check if a word is a content word: nouns, verbs, adj, adv.
    No auxiliaries (solved by a hack). No 's, be, do, or have lemmas."""
    # Penn Treebank tags: restrict content words to a selected subset of
    # nouns, verbs, adjectives, and adverbs (matching the original analysis).
    ContentTags = ['NN', 'NNS', 'NNP', 'NNPS', 'VB',
                   'VBD', 'VBG', 'VBN', 'VBZ', 'JJ', 'RB', 'VBP']
    if pos not in ContentTags:
        return False
    if pos.startswith("VB"): # hack: if that word is an aux anywhere in this sentence, it is not content
        # Remove verbs used as auxiliaries in any dependency of the sentence.
        auxes = [j.split(", ")[1].split("-")[0].lower() for j in deps if j.startswith("aux")]
        if word in auxes:
            return False
        pos = get_wordnet_pos(pos)
        if WordNetLemmatizer().lemmatize(get_lemma(word, pos)) in ["'s", "be", "do", "have"]:
            return False
    return True

def get_wordnet_pos(pos):
    """Map a Penn Treebank POS tag to the WordNet lemmatizer tag."""
    if pos.startswith("JJ"):
        return "a"
    if pos[0].lower() in ["n", "v", "a", "r"]:
        return pos[0].lower()
    return None

def get_lemma(word, pos):
    """Return a WordNet lemma where an appropriate POS mapping exists."""
    wordnet_pos = get_wordnet_pos(pos)
    lemma = word
    if wordnet_pos is not None:
        lemma = WordNetLemmatizer().lemmatize(word, pos=get_wordnet_pos(pos))
    return lemma

def get_word_info(X, Y, isSwitchboard):
    """Extract tokens, POS-prefixed lemmas, and content-word flags.

    X is a list of typed-dependency strings and Y contains word/POS pairs.
    The final whitespace-delimited item is skipped by the original code.
    ``isSwitchboard`` is retained for interface compatibility but the
    filtering it once controlled is currently disabled.
    """
    wordtokens = []
    wordlemmas = []
    content_words = []
    for _, word_pos in enumerate(Y.split(" ")[:-1]):
        word = word_pos.split("/")[0].lower()
        pos = word_pos.split("/")[1]
        # Preserve all tokens; content vs. non-content is encoded separately.
        # The original conditional content-word filter was disabled.
        if True: #is_content_word(word, pos, X) or isSwitchboard:
            wordtokens += [word]
            wordlemmas += [pos[:2] + "_" + get_lemma(word, pos)]
            content_words += [is_content_word(word, pos, X)]
    return wordlemmas, wordtokens, content_words

def parse_dep(s):
    """Parse a Stanford typed dependency into relation/head/dependent fields.

    Expected example: ``nsubj(runs-2, child-1)``. This implementation assumes
    hyphens only separate token text from token indices.
    """
    role_1, role_2 = s.split(", ")
    gramrole, role_3 = role_1.split("(")
    word, wordpos = role_2.rstrip(")").split("-")
    head, headpos = role_3.split("-")
    return {"gramrole": gramrole, "word": word.lower(),
            "wordpos": int(wordpos.rstrip("'")), "head": head, "headpos": int(headpos.rstrip("'"))}

def get_sent_dict(X, Y):
    """Attach POS tags from the parser token list to dependency entries.

    The mapping relies on the original ordering/format of Stanford output.
    """
    ds = []
    for i in X: 
        d = parse_dep(i)
        ds += [d]
    wordset = [j["word"] for j in ds]

    # Split only at the final slash, since a token itself may contain "/".
    pos = [p.rsplit("/", 1) for p in Y.split(" ") if p.rsplit("/", 1)[0].lower() in wordset]

    prevwordpos = 0
    for i, item in enumerate(ds):
        if item["wordpos"] > prevwordpos:
            p = pos.pop(0)
        ds[i]["wordfrompos"] = p[0]
        ds[i]["partofspeech"] = p[1]
        prevwordpos = item["wordpos"]

    return ds

def combine_headdict_and_pos(h, w):
    """Build a syntax-rule string for each non-root dependency head.

    Rule elements are dependency labels (for children) and the head POS tag,
    ordered by their positions in the sentence and joined by " + ".
    """
    output = []
    for i in h:
        if i != 0:
            relations = [j for j in h[i]]
            relations += [(i, w[i])]
            relations = sorted(relations, key=lambda x: x[0])
            relations = " + ".join([j[1] for j in relations])
            output += [relations]
    return output

def  simplify_pos(pos):
    """Collapse selected fine-grained POS tags into NOUN/VERB/ADJ/ADV.

    Tags not in the mapping remain unchanged (original analysis behavior).
    """
    try:
        return {'NNS': 'NOUN',
        'NNP': 'NOUN',
        'NNS': 'NOUN',
        'NN': 'NOUN',
        'RBR': 'ADV',
        'RBS': 'ADV',
        'WRB': 'ADV',
        'RBS': 'ADV',
        'RB': 'ADV',
        'JJR': 'ADJ',
        'JJS': 'ADJ',
        'JJ': 'ADJ',
        'VBZ': 'VERB',
        'VBD': 'VERB',
        'VBN': 'VERB',   
        'VBG': 'VERB',   
        'VBP': 'VERB',
        'VB': 'VERB'}[pos]
    except KeyError:
        return pos


def get_poslist(sd):
    """Return simplified POS tags, counting each new dependent position once."""
    curpos = 0
    poslist = []
    for word in sd:
        if word["wordpos"] > curpos:
            curpos = word["wordpos"]
            poslist += [simplify_pos(word["partofspeech"])]
    return poslist

def get_noun_verb_ratio(poslist):
    """Compute verbs / (nouns + verbs), or an empty value if undefined."""
    nouns = [i for i in poslist if i == "NOUN"]
    verbs = [i for i in poslist if i == "VERB"]
    if (len(nouns) + len(verbs)) > 0:
        return round(len(verbs)/(len(nouns) + len(verbs)), 2)
    return "" 

def make_pos_bigrams(poslist):
    """Generate adjacent POS pairs after prepending a <START> marker."""
    bigrams = nltk.bigrams(["<START>"] + poslist)
    return bigrams

def get_bigram_entropy(pos_bigrams):
    """Calculate conditional POS-bigram entropy pooled over one subject.

    Uses natural logarithms, so entropy is measured in nats. The start token
    is included; no end-of-sentence marker is appended.
    """
    pos_bigram_dict = defaultdict(Counter)
    # Total bigram count is used to weight each preceding-POS context.
    c = len(pos_bigrams)
    for bigram in pos_bigrams:
        pos_bigram_dict[bigram[0]][bigram[1]] += 1
    entropy = 0
    for k in pos_bigram_dict:
        for j in pos_bigram_dict[k]:
            entropy -= (pos_bigram_dict[k][j]/c) * \
                np.log(pos_bigram_dict[k][j]/sum(pos_bigram_dict[k].values()))
    return entropy

def get_dep_info(sd):
    """Compute open-arc maximum, mean dependency distance, and nsubj count.

    Root arcs are excluded from distance/open-arc calculations. Subject
    dependencies, including labels beginning with ``nsubj``, count as a
    proxy for clause count.
    """
    depc = Counter()
    depdists = []
    nsubj_count = 0
    for sent_dict in sd:
        if sent_dict["gramrole"] != "root":
            earlier = min([sent_dict["headpos"], sent_dict["wordpos"]])
            later = max([sent_dict["headpos"], sent_dict["wordpos"]])
            depdists += [abs(earlier - later)]
            for i in range(earlier, later):
                depc[i] += 1
        if sent_dict["gramrole"].startswith("nsubj"):
            nsubj_count += 1
    if len(depc) == 0:
        maxdep = ""
    else:
        maxdep = max(depc.values())
    if len(depdists) == 0:
        avg_dep = ""
    else:
        avg_dep = np.round(np.mean(depdists), 2)
    return maxdep, avg_dep, nsubj_count

def get_syntax_info(X, Y):
    """Return syntax-rule occurrences, sentence metrics, and POS bigrams."""
    head_dict = defaultdict(list)
    word_parts_of_speech = {}
    sent_dicts = get_sent_dict(X, Y)
    for sent_dict in sent_dicts:
        head_dict[sent_dict["headpos"]] += [(sent_dict["wordpos"], sent_dict["gramrole"])]
        word_parts_of_speech[sent_dict["wordpos"]] = simplify_pos(sent_dict["partofspeech"])
    poslist = get_poslist(sent_dicts)
    noun_verb_ratio = get_noun_verb_ratio(poslist)
    maxdep, avg_dep, nsubj_count = get_dep_info(sent_dicts)
    sent_alternative_metrics = {"length": len(poslist),
                                      "maxdep":  maxdep,
                                      "avg_dist": avg_dep,
                                      "noun_verb_ratio": noun_verb_ratio,
                                      "nsubj_count": nsubj_count}
    return (combine_headdict_and_pos(head_dict, word_parts_of_speech),
            sent_alternative_metrics,
            make_pos_bigrams(poslist))

def get_dict_for_person(person):
    """Read a subject's parser output and collect sentence/subject features.

    The parser output is assumed to contain six blank-line-separated blocks
    per sentence; selected blocks hold words/POS and typed dependencies.
    Returns token rows, syntax-rule rows, sentence metrics, and POS entropy.
    """
    word_list = []
    syntax_list = []
    alternative_metrics_list = []
    pos_bigrams_list = []
    # Stanford LexicalizedParser output is grouped into six sections here.
    x = open(f"Data/parses/{person}.txt").read().split("\n\n")
    sents = []
    for i in range(0, len(x), 6):
        try:
            sents += [(x[i], x[i + 1], x[i + 2])]
        except IndexError:
            pass
    for sentnum, sent in enumerate(sents):
        sentid = person + "_Sentence" + str(sentnum + 1)
        X = sent[2].split("\n")
        Y = sent[1]
        lemma_freqs, content_freqs, content_words = get_word_info(X, Y, is_switchboard(person))
        for wordnum in range(len(lemma_freqs)):
            word_list += [{"SubjectID": person,
                           "SentenceID": sentid,
                           "ContentWordLemmatized_original": lemma_freqs[wordnum],
                           "ContentWordToken": content_freqs[wordnum],
                           "IsContentWord": content_words[wordnum]}]
        syntax_rules, alternative_metrics, posbigrams = get_syntax_info(X, Y)
        for syntaxrule in syntax_rules:
            syntax_list += [{"SubjectID": person,
                             "SentenceID": sentid,
                             "SyntaxRule_original": syntaxrule}]
        alternative_metrics["SubjectID"] = person
        alternative_metrics["SentenceID"] = sentid
        alternative_metrics_list += [alternative_metrics]
        pos_bigrams_list += posbigrams
    # Unlike the other outputs, entropy is summarized at the subject level.
    pos_entropy = {"SubjectID": person, "POSentropy": get_bigram_entropy(pos_bigrams_list)}
    return word_list, syntax_list, alternative_metrics_list, pos_entropy


if __name__ == "__main__":
    # Process every entry under Data/parses (expected: <SubjectID>.txt).
    # Run from the repository root so relative Data/ paths resolve.
    words_d = []
    syntax_d = []
    alternative_metrics_list = []
    pos_entropy_list = []
    for person in os.listdir("Data/parses"):
        words, syntax, alternative_metrics, pos_entropy = get_dict_for_person(person.split(".txt")[0])
        words_d += [pd.DataFrame(words)]
        syntax_d += [pd.DataFrame(syntax)]
        alternative_metrics_list += [pd.DataFrame(alternative_metrics)]
        pos_entropy_list += [pd.DataFrame(pos_entropy, index=['SubjectID'])]
    words_d = pd.concat(words_d)
    # Preserve the source CSV layout, including pandas' default index column.
    words_d.loc[words_d["IsContentWord"] == True].to_csv("Data/wordsrules_content.csv")
    words_d.loc[words_d["IsContentWord"] == False].to_csv(
        "Data/wordsrules_nocontent.csv")
    pd.concat(syntax_d).to_csv("Data/syntaxrules.csv")
    pd.concat(alternative_metrics_list).to_csv("Data/alternative_metrics.csv")
    pd.concat(pos_entropy_list).to_csv("Data/pos_entropy.csv")
