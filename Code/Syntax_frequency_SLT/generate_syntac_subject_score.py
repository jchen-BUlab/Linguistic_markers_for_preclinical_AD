"""Calculate subject-level syntax scores from extracted syntactic rules.

Run after ``get_rule_counts.py`` has generated ``Data/syntaxrules.csv``.
The output is consumed by ``feature_generation.py`` and
``feature_generation_filter.py``.
"""

import math
import os
from pathlib import Path

import pandas as pd


def main():
    # Resolve defaults relative to this script so execution does not depend
    # on the shell's working directory. Environment variables can override them.
    script_dir = Path(__file__).resolve().parent
    syntaxrules_file = Path(
        os.environ.get(
            "FHS_SYNTAX_RULES_FILE", str(script_dir / "Data" / "syntaxrules.csv")
        )
    )
    feat_dir = Path(
        os.environ.get("FHS_FEATURE_DIR", str(script_dir.parent / "features"))
    )
    output_file = feat_dir / "syntac_subject_score.csv"

    syntaxrules = pd.read_csv(syntaxrules_file)

    # Count each syntax rule across the supplied file and use the original
    # log(count) score (without adding a pseudocount or filtering subjects).
    syn_dic = {}
    for s in syntaxrules["SyntaxRule_original"].unique():
        syn_dic[s] = 0
    for k in syntaxrules["SyntaxRule_original"].to_list():
        syn_dic[k] += 1
    for key in syn_dic.keys():
        syn_dic[key] = math.log(syn_dic[key])

    subjectID = syntaxrules["SubjectID"].unique()
    syntac_dict = {}

    for i in subjectID:
        sub_df = syntaxrules.loc[syntaxrules["SubjectID"] == i, :]

        # Preserve the original order of sentence IDs per subject.
        sid_ls = []
        for sentence in sub_df["SentenceID"].to_list():
            sid = sentence.split("_Sentence")[1]
            if sid not in sid_ls:
                sid_ls.append(sid)

        # Average the rule scores within each sentence, then average the
        # sentence scores for the subject. Sentences are equally weighted.
        total_score = 0
        for k in sid_ls:
            sub_sentence_df = sub_df.loc[
                sub_df["SentenceID"] == i + "_Sentence" + k, :
            ]
            sentence_score = 0
            for j in sub_sentence_df["SyntaxRule_original"].to_list():
                sentence_score += syn_dic[j]
            sentence_score_avg = sentence_score / len(
                sub_sentence_df["SyntaxRule_original"].to_list()
            )
            total_score += sentence_score_avg

        subject_score_avg = total_score / len(sid_ls)
        syntac_dict[i] = subject_score_avg

    data = [{"subjectID": k, "syntac_score": v} for k, v in syntac_dict.items()]
    syntac_score_df = pd.DataFrame(data)

    feat_dir.mkdir(parents=True, exist_ok=True)
    # Retain pandas' default CSV index to match the original notebook output.
    syntac_score_df.to_csv(output_file)
    print(f"Saved subject-level syntax scores: {output_file}")


if __name__ == "__main__":
    main()
