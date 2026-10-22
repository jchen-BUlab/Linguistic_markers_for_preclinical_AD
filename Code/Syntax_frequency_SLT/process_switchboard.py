"""Clean Switchboard transcripts for reference-corpus feature extraction.

Reads per-file transcripts from Data/Switchboard_Full and a small reference
transcript from Data/switchboard_clean_small.txt. Writes one cleaned text
file per large-corpus input to Data/rawtext, prefixed with "B", and appends
the cleaned small reference transcript to Data/rawtext/Switchboard_Small.txt.

The cleaning heuristics below are intentionally retained from the original
analysis. This is separate from the cleaning used by write_parser_commands.py.
"""

import os
import re
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer

# Common fillers/acknowledgments removed before dependency parsing.
DISFLUENCIES = ["uh-huh", "huh", "uh", "yeah", "hm", "oh", "-hum", "um",
                "yeah", "hi", "yep", "huh-uh", "hum-um"]
                
# Short stock utterances explicitly removed from the reference corpus.
BADSENTS = [" Really.",
"I.",
"I see.",
"I know.",
"That's right.",
"I think.",
"You know.",
"I guess.",
"It's.",
"That's.",
"Of course.",
"All right.",
"Bye-bye.",
"Thank you.",
"My gosh.",
"My god."]

# At sentence start, these fillers/connectives are stripped repeatedly.
BADSTARTS = DISFLUENCIES + ["okay",
"well",
"and",
"but",
"no",
"yes",
"sure",
"right",
"so"]


def clean_sent(words):
    """Remove selected disfluencies, adjacent repeats, and short utterances.

    Takes a mutable list of tokens and returns a cleaned sentence or None.
    Caution: the original ``while words[0]`` assumes the token list remains
    non-empty while removing leading fillers.
    """
    newlist = []
    prevword = ""
    # Strip leading fillers even if several occur consecutively.
    while words[0].lower() in BADSTARTS:
        words.pop(0)
    for word in words:
        # Consecutive repeated tokens are removed, not arbitrary repeats.
        if word != prevword:
            if word == "i":
                newlist += ["I"]
            elif word not in DISFLUENCIES:
                newlist += [word]
        prevword = word
    # Require at least three retained tokens before keeping a sentence.
    if len(newlist) > 2:
        newlist = [re.sub("-hum", "", i) for i in newlist]
        s =  TreebankWordDetokenizer().detokenize(newlist)
        s =   s[0].upper() + s[1:]
        if s not in BADSENTS and len(s.split(" ")) > 0:
            return s
    return None


# Historical output option (not used by the current pipeline):
#outfile = open(f"Data/cleaned/Switchboard_Full_Clean.txt", "w")
#outfile.close()
# Convert every file in the full Switchboard corpus into a B-prefixed text.
for num in os.listdir("Data/Switchboard_Full"):
    a = [clean_sent(TreebankWordTokenizer().tokenize(i.strip()))
     for i in open(f"Data/Switchboard_Full/{str(num)}").readlines()]
    a = [i for i in a if i is not None]
    outfile = open(f"Data/rawtext/B{num}", "w")
    outfile.write("\n".join(a))
    outfile.close()

# Process the separate small Switchboard sample for validation/analysis.
a = [clean_sent(TreebankWordTokenizer().tokenize(i.strip()))
     for i in open("Data/switchboard_clean_small.txt").readlines()]
a = [i for i in a if i is not None]
# Append mode preserves existing output across runs; clear it manually if
# re-running the script should start with a clean reference file.
outfile = open(f"Data/rawtext/Switchboard_Small.txt", "a")
outfile.write("\n".join(a))
outfile.close()
