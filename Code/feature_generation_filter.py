"""Extract linguistic features after filtering specified noise-related utterances.

This variant excludes utterances containing an I/you subject linked to the
lemmas forget, remember, or recall from lexical and syntactic calculations.
Raw word totals, speech rates, noise measures, and pause measures retain their
original definitions; see the per-feature comments for the mixed denominators.
The source transcripts and subject-level metadata are not distributed here.
"""

import math
import os
import re

import pandas as pd
import spacy
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tree import Tree

from data_util import data_voice_clean_dir, feat_dir, output_dir


class linguisticFeatExtractor:
    """Extract subject-level lexical and syntax-derived language features."""

    def __init__(self, config_file=None):
        """Initialize NLTK utilities; config_file is kept for compatibility."""
        self.ptStemmer = PorterStemmer()
        self.WNL = WordNetLemmatizer()

    def preprocessing(self,text):
        """Remove bracketed annotations and return spaCy tokens and coarse POS tags.

        Returns (tokens, pos_ls), where pos_ls contains (token, POS) tuples.
        The original pipeline preserves casing and does not stem these tokens.
        """
        nlp = spacy.load("en_core_web_sm")
        new_text = re.sub(r"\[.*?\]", "", text)
        doc = nlp(new_text)
        tokens = [token.text for token in doc]
        pos_tags = [token.pos_ for token in doc]
        pos_ls = list(zip(tokens, pos_tags))
        return tokens, pos_ls

    def clean_text(self,seg_ls):
        """Remove CHAT-style transcript annotations and collect marked interjections.

        Returns (new_seg_ls, INTJ). The list INTJ is used to override POS tags
        for tokens originally preceded by the '&-' filled-pause marker.
        """
        new_seg_ls=[]
        INTJ=[]
        for seg in seg_ls:
            new_seg = re.sub(r'\[.*?\]', '', seg)
            new_seg = re.sub(r'<.*?>', '', new_seg)
            words=new_seg.strip().split()
            clean_seg=''
            for word in words:
                word=word.replace(':','')
                if '&-' in word:
                    word=word.replace('&-','')
                    INTJ.append(word)
                word=word.replace('^','')
                word=word.replace('(','')
                word=word.replace(')','')
                word=word.replace('[?]','')
                word=word.replace('*','')
                word=word.replace(',','')
                word=word.replace('!','')
                word=word.replace('≠','')
                word=re.sub(r'↫.*?↫','',word)
                if '&+' in word or '\\' in word or '/' in word or '&' in word:
                    word=''
                if len(word)>1 and '.' in word:
                    word=''
                if len(word)>0:
                    clean_seg=clean_seg+word+' '
            new_seg_ls.append(clean_seg)

        return new_seg_ls,INTJ

    # POS-category predicates used by lexical_features(). Keep names intact.
    def _isWords(self,pos):
        return pos not in ("PUNCT", "SYM")

    def _isNoun(self, pos):
        return pos == "NOUN"

    def _isVerb(self, pos):
        return pos == "VERB"

    def _isauxiliary(self, pos):
        return pos == "AUX"

    def _isAdjective(self, pos):
        return pos == "ADJ"

    def _isAdverb(self, pos):
        return pos == "ADV"

    def _isConjunciton(self, pos):
        return pos in ("CONJ", "CCONJ", "SCONJ")

    def _isPronoun(self, pos):
        return pos == "PRON"

    def _isPropernoun(self,pos):
        return pos == "PROPN"

    def _isDeterminer(self, pos):
        return pos == "DET"

    def _isInterjection(self, pos):
        return pos == "INTJ"

    def _isPreposition(self, pos):
        return pos == "ADP"

    def _isParticle(self, pos):
        return pos == "PART"

    def _isQuestion(self,token):
        return token == "?"

    def _isPause(self,token):
        return token == "."

    def _isOpenWords(self,pos):
        return pos in ("NOUN", "VERB", "ADJ", "ADV", "SYM")

    def _isSymbol(self,pos):
        return pos == "SYM"

    def func_meanWordLength(self, tokens):
        """Mean token length, excluding only literal '.' tokens (legacy rule)."""
        length=0
        number=0
        for word in tokens:
            if word != '.':
                length+=len(word)
                number+=1
        return length/number

    

    def calc_words(self,t):
        """Recursively count terminal tokens in a constituency tree."""
        if type(t) == str:
            return 1
        else:
            val = 0
            for child in t:
                val += self.calc_words(child)
            return val

    def calc_nodes(self,t):
        """Count child nodes recursively using the original tree convention."""
        if type(t) == str:
            return 0
        else:
            val = 0
            for child in t:
                val += self.calc_nodes(child)+1
            return val

    def calc_yngve(self,t, par):
        """Sum right-to-left Yngve stack-depth weights across tree leaves."""
        if type(t) == str:
            return par
        else:
            val = 0
            for i, child in enumerate(reversed(t)):
                val += self.calc_yngve(child, par+i)
            return val

    def is_sent(sefl,val):
        """Check whether a parse-node label begins with S."""
        return len(val) > 0 and val[0] == "S"
        
    def calc_frazier(self,leaf_scores):
        """Maximum sum of up to three consecutive leaf-level Frazier scores."""
        if len(leaf_scores) <=3:
            return sum(leaf_scores)
        max_sum=sum(leaf_scores[:3])
        current_sum=max_sum
        for i in range(3,len(leaf_scores)):
            current_sum=current_sum+leaf_scores[i]-leaf_scores[i-3]
            max_sum=max(max_sum,current_sum)
        return max_sum

    def calc_frazier_roark(self,t, par, par_lab,leaf_scores):
        """Recursively assign left-branching complexity scores to tree leaves."""
        if type(t) == str:
            leaf_score=par-1
            leaf_scores.append(leaf_score)
            return leaf_score,leaf_scores
        else:
            val = 0
            for i, child in enumerate(t):
                # For all but the leftmost child, zero
                score = 0
                if i == 0:
                    my_lab = t.label()
                    # If it's a sentence, and not duplicated, add 1.5
                    if self.is_sent(my_lab):
                        score = (0 if self.is_sent(par_lab) else par+1.5)
                    # Otherwise, unless it's a root node, add one
                    elif my_lab != "" and my_lab != "ROOT" and my_lab != "TOP":
                        score = par + 1
                child_val, leaf_scores = self.calc_frazier_roark(child, score, my_lab, leaf_scores)
                val += child_val
            return val , leaf_scores

    def calc_MDD(self,sentence):
        """Mean absolute token-to-head distance from spaCy dependency parses.

        Replicates the original calculation, including the ROOT self-link
        (distance zero) in the token count.
        """
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(sentence)
        total_dependency_distance = 0
        count = 0
        for token in doc:
            if token.head is not None:
                dependency_distance = abs(token.head.i - token.i)
                total_dependency_distance += dependency_distance
                count += 1
        mdd = total_dependency_distance / count if count else 0
        return mdd

    def calc_between_utterance_pause(self,df):
        """Sum the intervals after PAR rows; count intervals >= 2000 ms.

        Uses the next row of the supplied DataFrame, regardless of its tier.
        Input timestamps are assumed to be in milliseconds.
        """
        between_utterance_pause=0
        between_utterance_pause_number=0
        for index, row in df.iterrows():
            if  index < df.shape[0]-1:
                if df.loc[index,'tier']== 'PAR':
                    time=df.loc[index+1,'start_time']-df.loc[index,'end_time']
                    between_utterance_pause+=time
                    if time >=2000:
                        between_utterance_pause_number+=1

        return between_utterance_pause/1000, between_utterance_pause_number

    def lexical_features(self,feat_dict,pos_ls,totalUtterance,totalUtterance_filtered,tokens,sid,totalDurationTime,totalWords,totalUniqueWords,totalWords_filtered,totalUniqueWords_filtered):
        """Calculate lexical and POS features using the filtered utterances.

        Most POS and lexical ratios use totalWords_filtered; speechRate and
        uniqueSpeechRate intentionally use unfiltered word counts and time.
        Preserve the input's original mixture of filtered and raw denominators.
        """
        totalNouns = 0
        totalVerbs = 0
        totalAdjectives = 0
        totalAdverbs = 0
        totalPronouns = 0
        totalConjunctions = 0
        totalDeterminers = 0
        totalInterjections = 0
        totalPrepositions = 0
        totalParticles = 0
        totalOpenWords=0
        totalCloseWords=0
        totalPropernouns=0
        totalAuxiliary = 0
        Verbs_set=set()
        Nouns_set=set()
        Adjectives_set=set()
        Adverbs_set=set()
        Conjunctions_set=set()
        Determiners_set=set()
        Interjections_set=set()
        Prepositions_set=set()
        Particles_set=set()
        Pronouns_set=set()
        Propernouns_set=set()
        Auxiliary_set=set()

        for (tok, pos) in pos_ls:
            if self._isOpenWords(pos):
                totalOpenWords+=1
            else:
                totalCloseWords+=1
            if self._isPropernoun(pos):
                Propernouns_set.add(tok)
                totalPropernouns +=1
            if self._isNoun(pos):
                Nouns_set.add(tok)
                totalNouns += 1
            elif self._isAdverb(pos):
                Adverbs_set.add(tok)
                totalAdverbs+=1
            elif self._isVerb(pos):
                Verbs_set.add(tok)
                totalVerbs += 1
            elif self._isAdjective(pos):
                Adjectives_set.add(tok)
                totalAdjectives += 1
            elif self._isPronoun(pos):
                Pronouns_set.add(tok)
                totalPronouns += 1
            elif self._isConjunciton(pos):
                Conjunctions_set.add(tok)
                totalConjunctions += 1
            elif self._isDeterminer(pos):
                Determiners_set.add(tok)
                totalDeterminers += 1
            elif self._isInterjection(pos):
                Interjections_set.add(tok)
                totalInterjections += 1
            elif self._isPreposition(pos):
                Prepositions_set.add(tok)
                totalPrepositions +=1
            elif self._isParticle(pos):
                Particles_set.add(tok)
                totalParticles += 1
            elif self._isauxiliary(pos):
                Auxiliary_set.add(tok)
                totalAuxiliary+=1

        # Unique token counts are tracked separately for each POS category.
        totalUniqueVerbs = len(Verbs_set)
        totalUniqueNouns = len(Nouns_set)
        totalUniqueAdjectives = len(Adjectives_set)
        totalUniqueAdverbs =  len(Adverbs_set)
        totalUniqueConjunctions = len(Conjunctions_set)
        totalUniqueDeterminers = len(Determiners_set)
        totalUniqueInterjections = len(Interjections_set)
        totalUniquePrepositions = len(Prepositions_set)
        totalUniqueParticles = len(Particles_set)
        totalUniquePronouns = len(Pronouns_set)
        totalUniquePropernouns = len(Propernouns_set)
        totalUniqueAuxiliary = len(Auxiliary_set)

        # POS occurrence rates use only words in retained utterances.
        nounsRate = totalNouns/totalWords_filtered
        verbsRate = totalVerbs/totalWords_filtered
        adjectivesRate = totalAdjectives/totalWords_filtered
        adverbsRate = totalAdverbs/totalWords_filtered
        pronounsRate = totalPronouns/totalWords_filtered
        conjunctionsRate = totalConjunctions/totalWords_filtered
        determinersRate = totalDeterminers/totalWords_filtered
        interjectionsRate = totalInterjections/totalWords_filtered
        prepositionsRate = totalPrepositions/totalWords_filtered
        propernounsRate = totalPropernouns / totalWords_filtered
        particlesRate = totalParticles/totalWords_filtered
        auxiliaryRate = totalAuxiliary / totalWords_filtered
        # Unique POS-token rates use the same filtered word denominator.
        uniqueNounsRate = totalUniqueNouns/totalWords_filtered
        uniqueVerbsRate = totalUniqueVerbs/totalWords_filtered
        uniqueAdjectivesRate = totalUniqueAdjectives/totalWords_filtered
        uniqueAdverbsRate = totalUniqueAdverbs/totalWords_filtered
        uniquePronounsRate = totalUniquePronouns/totalWords_filtered
        uniqueConjunctionsRate = totalUniqueConjunctions/totalWords_filtered
        uniqueDeterminersRate = totalUniqueDeterminers/totalWords_filtered
        uniqueInterjectionsRate = totalUniqueInterjections/totalWords_filtered
        uniquePrepositionsRate = totalUniquePrepositions/totalWords_filtered
        uniquePropernounsRate = totalUniquePropernouns / totalWords_filtered
        uniqueParticlesRate = totalUniqueParticles/totalWords_filtered
        uniqueAuxiliaryRate = totalUniqueAuxiliary / totalWords_filtered

        typeTokenRatio=totalUniqueWords_filtered/totalWords_filtered
        # Speech rates keep the unfiltered totals by design in this variant.
        speechRate = totalWords/totalDurationTime
        uniqueSpeechRate= totalUniqueWords / totalDurationTime
        
        # POS-specific unique counts are summed (not a global token union).
        totalUniqueContentWords = totalUniqueNouns+totalUniqueVerbs+totalUniqueAdjectives+ totalUniqueAdverbs
        # Retain the supplied filtered script's use of the unfiltered totalUniqueWords.
        totalUniqueCloseWords= totalUniqueWords-totalUniqueContentWords

        contentDensity = totalOpenWords/(totalCloseWords+1)
        uniqueContentDensity = totalUniqueContentWords / (totalCloseWords+1)
        uniqueContentDensityUnique = totalUniqueContentWords / (totalUniqueCloseWords +1)

        pronounProperNounRatio= totalPronouns / (totalPropernouns+1)
        meanWordLength=self.func_meanWordLength(tokens)
        wordsPerSentence = totalWords_filtered / totalUtterance_filtered
        uniqueWordsPerSentence= totalUniqueWords_filtered / totalUtterance_filtered
        ideaDensity=(totalVerbs + totalAdjectives + totalAdverbs + totalPrepositions + totalConjunctions)/totalWords_filtered
        uniqueIdeaDensity = (totalUniqueVerbs+totalUniqueAdjectives+totalUniqueAdverbs+totalUniquePrepositions+totalUniqueConjunctions)/ totalWords_filtered

        feat_dict[sid]['totalWords'] = totalWords
        feat_dict[sid]['totalUniqueWords'] = totalUniqueWords
        feat_dict[sid]['totalWords_filtered'] = totalWords_filtered
        feat_dict[sid]['totalUniqueWords_filtered'] = totalUniqueWords_filtered
        feat_dict[sid]['nounsRate'] = nounsRate
        feat_dict[sid]['verbsRate'] = verbsRate
        feat_dict[sid]['adjectivesRate'] = adjectivesRate
        feat_dict[sid]['adverbsRate'] = adverbsRate
        feat_dict[sid]['pronounsRate'] = pronounsRate
        feat_dict[sid]['conjunctionsRate'] = conjunctionsRate
        feat_dict[sid]['determinersRate'] = determinersRate
        feat_dict[sid]['interjectionsRate'] = interjectionsRate
        feat_dict[sid]['prepositionsRate'] = prepositionsRate
        feat_dict[sid]['propernounsRate'] = propernounsRate
        feat_dict[sid]['particlesRate'] = particlesRate
        feat_dict[sid]['auxiliaryRate'] = auxiliaryRate
        feat_dict[sid]['uniqueNounsRate'] = uniqueNounsRate
        feat_dict[sid]['uniqueVerbsRate'] = uniqueVerbsRate
        feat_dict[sid]['uniqueAdjectivesRate'] = uniqueAdjectivesRate
        feat_dict[sid]['uniqueAdverbsRate'] = uniqueAdverbsRate
        feat_dict[sid]['uniquePronounsRate'] = uniquePronounsRate
        feat_dict[sid]['uniqueConjunctionsRate'] = uniqueConjunctionsRate
        feat_dict[sid]['uniqueDeterminersRate'] = uniqueDeterminersRate
        feat_dict[sid]['uniqueInterjectionsRate'] = uniqueInterjectionsRate
        feat_dict[sid]['uniquePrepositionsRate'] = uniquePrepositionsRate
        feat_dict[sid]['uniquePropernounsRate'] = uniquePropernounsRate
        feat_dict[sid]['uniqueParticlesRate'] = uniqueParticlesRate
        feat_dict[sid]['uniqueAuxiliaryRate'] = uniqueAuxiliaryRate
        feat_dict[sid]['typeTokenRatio'] = typeTokenRatio
        feat_dict[sid]['speechRate'] = speechRate
        feat_dict[sid]['uniqueSpeechRate'] = uniqueSpeechRate
        feat_dict[sid]['contentDensity'] = contentDensity
        feat_dict[sid]['uniqueContentDensity']=uniqueContentDensity
        feat_dict[sid]['uniqueContentDensityUnique']=uniqueContentDensityUnique
        feat_dict[sid]['pronounProperNounRatio'] = pronounProperNounRatio
        feat_dict[sid]['meanWordLength'] = meanWordLength
        feat_dict[sid]['wordsPerSentence'] = wordsPerSentence
        feat_dict[sid]['uniqueWordsPerSentence'] = uniqueWordsPerSentence
        feat_dict[sid]['ideaDensity'] = ideaDensity
        feat_dict[sid]['uniqueIdeaDensity'] = uniqueIdeaDensity
        feat_dict[sid]['totalDurationTime'] = totalDurationTime

        return feat_dict

    def syntactic_features(self,ut_ls,parser,feat_dict,sid):
        """Extract constituency-tree complexity and dependency distance.

        Each utterance contributes one observation to the mean, including
        zero scores when no constituency parse is available. For multi-sentence
        utterances, the original code uses the last constituency tree.
        """
        sents = 0
        yngve_tot = 0
        frazier_tot = 0
        frazier_roark_tot=0
        nodes_tot = 0
        mdd_tot=0
        for k in ut_ls:
            tree_object=None
            leaf_scores=[]
            doc=parser(k)
            for sent in doc.sents:
                if not hasattr(sent._, 'parse_string'):
                    pass
                else:
                    tree = sent._.parse_string
                    tree_object = Tree.fromstring(tree)
            if tree_object == None:
                words=0
                yngve=0
                yngve_avg=0
                nodes_avg=0
                nodes=0
                frazier=0
                frazier_roark=0
                frazier_roark_avg=0
                mdd=0
            else:
                t = tree_object
                words = self.calc_words(t)
                yngve = self.calc_yngve(t, 0)
                yngve_avg = float(yngve)/words
                nodes = self.calc_nodes(t)
                nodes_avg=float(nodes)/words
                frazier_roark,leaf_scores= self.calc_frazier_roark(t, 0, "",leaf_scores)
                frazier=self.calc_frazier(leaf_scores)
                frazier_roark_avg = float(frazier_roark)/words
                mdd=self.calc_MDD(k)
            sents += 1
            yngve_tot += yngve_avg
            nodes_tot += nodes_avg
            frazier_tot+=frazier
            frazier_roark_tot += frazier_roark_avg
            mdd_tot += mdd
        yngve_avg = float(yngve_tot)/sents
        frazier_avg=float(frazier_tot)/sents
        frazier_roark_avg = float(frazier_roark_tot)/sents
        nodes_avg = float(nodes_tot)/sents
        mdd_avg = float(mdd_tot)/sents
        feat_dict[sid]['yngve_avg']=yngve_avg
        feat_dict[sid]['frazier_avg']=frazier_avg
        feat_dict[sid]['frazier_roark_avg']=frazier_roark_avg
        feat_dict[sid]['nodes_avg']=nodes_avg
        feat_dict[sid]['mdd_avg']=mdd_avg
        
        return feat_dict

    def sentence_noise_calculate(self, seg, nlp,):
        """Count sentences with a target memory verb and I/you subject."""
        new_seg = re.sub(r'\[.*?\]', '', seg)
        doc = nlp(new_seg)
        subj_targets = {"i", "you"}
        verb_targets = {"forget", "remember", "recall"}

        sentence_count = 0

        for sent in doc.sents:
            lemmas = [token.lemma_.lower() for token in sent]
            has_target_verb = any(verb in lemmas for verb in verb_targets)
            if not has_target_verb:
                continue
            has_target_subj = False
            for token in sent:
                if token.dep_ in ("nsubj", "nsubjpass") and token.lemma_.lower() in subj_targets:
                    has_target_subj = True
                    break

            if has_target_subj:
                sentence_count += 1

        return sentence_count

    def calculate_noise(self, seg, nlp):
        """Count lemmas I/you/forget/remember/recall (legacy noise metric)."""
        new_seg = re.sub(r'\[.*?\]', '', seg)

        doc = nlp(new_seg)
        lemmas = [token.lemma_.lower() for token in doc]
        target_words = ["i", "you", "forget", "remember", "recall"]
        totalNoise = sum(lemmas.count(word) for word in target_words)

        return totalNoise

    def calculate_totalWords(self,seg,nlp,tok_set):
        """Count non-punctuation/non-symbol tokens; update unique-token set."""
        words=0
        new_seg = re.sub(r'\[.*?\]', '', seg)
        # Tokenization
        doc=nlp(new_seg)
        tokens=[token.text for token in doc]
        pos_tags=[token.pos_ for token in doc]
        pos_ls=list(zip(tokens, pos_tags))
        for token,pos in pos_ls:
            if pos != 'PUNCT' and pos !='SYM':
                words+=1
                tok_set.add(token)
        return words,tok_set

    def calc_filled_pause(self,seg):
        """Count '&-' filled-pause markers in the provided utterances."""
        total_filled_pause = 0
        for item in seg:
            words = item.strip().split()
            for word in words:
                if '&-' in word:
                    total_filled_pause += 1
        return total_filled_pause

    def calc_within_pause(self,seg):
        """Count '(.)' within-utterance pause markers in the input segments."""
        total_within_pause = 0
        for item in seg:
            words = item.strip().split()
            for word in words:
                if '(.)' in word:
                    total_within_pause += 1
        return total_within_pause

    def filter_tokens_and_pos(self,tokens, pos_ls):
        """Exclude SPACE and PUNCT from token/POS lists; retain SYM."""
        clean_tokens = []
        clean_pos_ls = []
        
        for token, pos in pos_ls:
            if pos not in ['SPACE', 'PUNCT']: 
                clean_tokens.append(token)
                clean_pos_ls.append((token, pos)) 

        return clean_tokens, clean_pos_ls


def gather_files(indir):
    """List inputs in the original os.walk order without filtering filenames."""
    file_info_ls = []
    for r, d, f in os.walk(indir):
        for filename in f:
            infile = os.path.join(r, filename)
            file_info_ls.append(infile)
    return file_info_ls


def main():
    """Generate subject-level features from retained participant utterances.

    Inputs require ID, tier, contents, duration, start_time and end_time.
    A sentence is excluded when the original sentence_noise_calculate()
    returns a nonzero count; raw totals and noise counts are kept separately.
    The original last-ID assignment and math.log() behavior are preserved.
    """
    indir = data_voice_clean_dir
    file_ls = gather_files(indir)
    feat_df = pd.DataFrame()
    syntac_score_df = pd.read_csv(
        os.path.join(feat_dir, "syntac_subject_score.csv")
    )  # Prior syntax-frequency scores produced by the separate module.

    lingExt = linguisticFeatExtractor()
    nlp = spacy.load("en_core_web_sm")
    import benepar  # Registers the benepar spaCy pipeline component.
    if "benepar" not in nlp.pipe_names:
        nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    log_noise_sentence = []
    for infile in file_ls:
        feat_dict = {}
        df = pd.read_csv(infile)
        sid_ls = df["ID"].unique()

        for sid in sid_ls:
            feat_dict[sid] = {}
            text = ""
            # Keep the original participant-only filtering and duration rule.
            seg_ls = df[(df["ID"] == sid) & (df["tier"] == "PAR")]["contents"]
            duration = df[(df["ID"] == sid) & (df["tier"] == "PAR")]["duration"]
            totalDurationTime = duration.sum() / 1000
            totalUtterance = seg_ls.shape[0]
            totalNoise = 0
            totalNoiseSentences = 0
            totalWords = 0
            totalWords_filtered = 0

            # Restrict pause intervals to rows belonging to this ID.
            # Adjacent rows are determined within df_sid, not the full CSV.
            df_sid = df.loc[df["ID"] == sid].reset_index(drop=True)
            betweenUtterancePause, betweenUtterancePauseNumber = (
                lingExt.calc_between_utterance_pause(df_sid)
            )
            totalFilledPauses = lingExt.calc_filled_pause(seg_ls)
            withinUtterancePause = lingExt.calc_within_pause(seg_ls)

            new_seg_ls, INTJ_list = lingExt.clean_text(seg_ls)
            tok_set = set()
            tok_set_unique = set()
            filt_seg = []
            for seg in new_seg_ls:
                totalNoise += lingExt.calculate_noise(seg, nlp)
                # Identify memory-related first/second-person statements.
                # The same count determines both the log and the exclusion.
                noise_sentence_count = lingExt.sentence_noise_calculate(seg, nlp)
                totalNoiseSentences += noise_sentence_count
                if noise_sentence_count == 1:
                    log_noise_sentence.append({
                        "infile": os.path.basename(infile),
                        "sid": sid,
                        "category": "noise",
                        "segment": seg.strip(),
                    })
                # Count all words before excluding noise-related utterances.
                totalWord, tok_set = lingExt.calculate_totalWords(seg, nlp, tok_set)
                totalWords += totalWord
                if noise_sentence_count == 0:
                    # Only retained segments contribute to filtered lexical
                    # and syntactic features, and to their word denominators.
                    filt_seg.append(seg)
                    text += " " + seg.strip("\n")
                    totalWord_filtered, tok_set_unique = lingExt.calculate_totalWords(
                        seg, nlp, tok_set_unique
                    )
                    totalWords_filtered += totalWord_filtered
                else:
                    continue

            noiseRate = totalNoise / totalWords
            totalUniqueWords = len(tok_set)
            totalUniqueWords_filtered = len(tok_set_unique)

            seg_pure = []
            for seg in filt_seg:
                seg = seg.replace(".", "").replace("?", "").replace("-", "")
                seg = seg.replace("]", "").strip()
                seg = re.sub(r"\s+", " ", seg)
                seg_pure.append(seg)

            # Parsing and POS tagging are performed on retained segments only.
            tokens, pos_ls = lingExt.preprocessing(text)
            tokens, pos_ls = lingExt.filter_tokens_and_pos(tokens, pos_ls)
            totalUtterance_filtered = len(filt_seg)
            pos_ls_updated = []
            for pos in pos_ls:
                pos = list(pos)
                if pos[0] in INTJ_list:
                    pos[1] = "INTJ"
                pos = tuple(pos)
                pos_ls_updated.append(pos)

            feat_dict = lingExt.lexical_features(
                feat_dict=feat_dict, pos_ls=pos_ls_updated,
                totalUtterance=totalUtterance,
                totalUtterance_filtered=totalUtterance_filtered,
                tokens=tokens, sid=sid,
                totalDurationTime=totalDurationTime, totalWords=totalWords,
                totalUniqueWords=totalUniqueWords,
                totalWords_filtered=totalWords_filtered,
                totalUniqueWords_filtered=totalUniqueWords_filtered,
            )
            feat_dict = lingExt.syntactic_features(
                ut_ls=seg_pure, parser=nlp, feat_dict=feat_dict, sid=sid
            )

            subject_ID = os.path.basename(infile).split(".csv")[0]
            # Match by transcript basename, preserving the existing join logic.
            feat_dict[sid]["syntactic_score"] = syntac_score_df.loc[
                syntac_score_df["subjectID"] == subject_ID, "syntac_score"
            ].values[0]
            feat_dict[sid]["subjectID"] = subject_ID
            feat_dict[sid]["totalNoise"] = totalNoise

        # Original workflow: these assignments were outside the ID loop.
        # They therefore populate only the final ID in each input CSV.
        # Preserve this behavior for exact historical output compatibility.
        feat_dict[sid]["logBetweenUtterancePause"] = math.log(betweenUtterancePause)
        feat_dict[sid]["betweenUtterancePauseNumber"] = betweenUtterancePauseNumber
        feat_dict[sid]["totalFilledPauses"] = totalFilledPauses
        feat_dict[sid]["withinUtterancePause"] = withinUtterancePause
        feat_dict[sid]["noiseRate"] = noiseRate
        feat_dict[sid]["totalNoiseSentences"] = totalNoiseSentences
        # Historical definition: noise-sentence count / retained segment count.
        # A zero retained-segment count still raises ZeroDivisionError, as in
        # the supplied filtered version (do not silently alter outputs).
        feat_dict[sid]["noiseSentencesRate"] = totalNoiseSentences / len(filt_seg)
        if (withinUtterancePause + totalFilledPauses + betweenUtterancePauseNumber) == 0:
            feat_dict[sid]["standardizedPauseRate"] = totalWords
        else:
            feat_dict[sid]["standardizedPauseRate"] = totalWords / (
                withinUtterancePause + totalFilledPauses + betweenUtterancePauseNumber
            )

        df_temp = pd.DataFrame.from_dict(feat_dict, orient="index")
        feat_df = pd.concat([feat_df, df_temp], axis=0)

    os.makedirs(output_dir, exist_ok=True)
    feat_df.to_csv(
        os.path.join(output_dir, "linguistic_feature_spacy_clean_filtered_20251116.csv"),
        index=False,
    )
    # This log includes transcript excerpts and must not be committed to GitHub.
    log_df = pd.DataFrame(log_noise_sentence)
    log_df.to_csv(
        os.path.join(output_dir, "selected_noise_segments_20251116_filtered.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
