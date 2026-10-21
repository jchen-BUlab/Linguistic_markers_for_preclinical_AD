"""Clean per-subject text and generate Stanford parser shell commands.

Configuration uses portable relative defaults for Data/ and Stanford
CoreNLP 2018-10-05. Set environment variables to override the defaults.
Input:  <DATA_DIR>/clean_data_raw_txt/<subject file>
Output: <DATA_DIR>/cleaned/<subject file>, plus Data/run_parses0.sh etc.

This script WRITES parser commands; it does not execute them. The generated
commands assume they are run from the Data directory (cleaned/ and parses/
are relative paths in the shell commands).
"""

import os
import random

# Portable configuration: override for a different project/data location.
# The classpath wildcard includes the Stanford CoreNLP JAR files.
p = os.path.join(os.environ.get("SYNTAX_DATA_DIR", "Data"), "")
stanford_path = os.environ.get(
    "STANFORD_CORENLP_CLASSPATH", "../stanford-corenlp-full-2018-10-05/*"
)
# Subject-level raw transcripts awaiting cleaning/parsing.
y = os.listdir(p + "clean_data_raw_txt/")
scripts = []
def read_in_txtdata(inputfile, temp_filename):
    """Apply the original transcript-cleaning substitutions per line.

    Removes selected fillers/leading discourse words, normalizes terminal
    punctuation, and writes one cleaned sentence per line. This heuristic
    cleaner is distinct from the Switchboard preprocessing script.
    """
    tf2 = []
    tf3 = []
    with open(inputfile, "r+") as infile:
        tf2 = [i.strip() for i in infile.readlines()]
    # Keep the replacement order exactly as in the original analysis.
    for i in range(len(tf2)):
        temp = tf2[i].replace('Uh ', '')
        temp = temp.replace('-', '')
        temp = temp.replace('Uhh ', '')
        temp = temp.replace('uhh ', '')
        temp = temp.replace('And ', '')
        temp = temp.replace('But ', '')
        temp = temp.replace('Well, ', '')
        temp = temp.replace('Well ', '')
        temp = temp.replace('Ok ', '')
        temp = temp.replace('Ok, ', '')
        temp = temp.replace('Okay, ', '')
        temp = temp.replace('Okay ', '')
        temp = temp.replace('Yeah, ', '')
        temp = temp.replace('Yeah ', '')
        temp = temp.replace(' oh oh ', ' ')
        temp = temp.replace(' oh ', ' ')
        temp = temp.replace('And and ', '')
        temp = temp.replace('Or ', '')
        temp = temp.replace('Oh ', '')
        temp = temp.replace('And uhh ', '')
        temp = temp.replace('uhh uhh ', '')
        temp = temp.replace('Uhh uhh ', '')
        temp = temp.replace('Uhh and uhh ', '')
        temp = temp.replace('And then ', '')
        temp = temp.replace('!', '.')
        if len(temp)!=0:
            if temp[-1]  not in [".", "?", "!"]:
                temp += "."
            if tf2[i][0] != '(' and tf2[i] != '.':
                tf3.append(temp)
    # Output is consumed as one sentence per line by Stanford Parser.
    with open(temp_filename, 'w') as outfile:
        for i in range(len(tf3)):
            outfile.write(tf3[i][0].upper() + tf3[i][1:] + '\n')

if __name__ == "__main__":
    # First clean all files, then emit shell commands for batch parsing.
    for t in y:
        read_in_txtdata(os.path.join(p, "clean_data_raw_txt", t),
                        os.path.join(p, "cleaned", t))
        # Output format must match the six-block parser layout expected by
        # get_rule_counts.py (words, dependencies, and word/POS tags).
        scripts += [f"""java -mx800m -cp "{stanford_path}" edu.stanford.nlp.parser.lexparser.LexicalizedParser -outputFormat "words,typedDependenciesCollapsed,wordsAndTags" -printPCFGkBest 1 -sentences newline edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz  cleaned/{t} > parses/{t}"""]

    # Spread files across independent shell scripts for parallel execution.
    # No seed is set; script assignments can differ between runs.
    random.shuffle(scripts)
    scripts_total = [scripts[:110], scripts[110:220],
                    scripts[220:330], scripts[330:440],
                    scripts[440:550],scripts[550:]]
    # Corrected an original Python syntax typo here: [440:550] -> scripts[440:550].

    # These files are written to Data/, but the *commands inside* use paths
    # relative to Data/; see README for the correct working directory.
    for i, script in enumerate(scripts_total):
        outfile = open(f"Data/run_parses{str(i)}.sh", "w")
        outfile.write("\n".join(script))
        outfile.close()
