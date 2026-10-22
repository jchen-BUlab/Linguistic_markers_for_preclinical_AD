# Syntax and Lexical Frequency Analysis

Scripts for extracting syntactic and lexical features from speech transcripts and analyzing syntax–lexicon relationships. The workflow uses the Stanford Parser to identify dependency-based syntactic rules and the Switchboard corpus as a reference for word and syntax frequencies. The complete `SLT.R` analysis script is included.

## Files

| File | Description |
| --- | --- |
| `process_switchboard.py` | Cleans Switchboard transcripts for reference-corpus processing. |
| `write_parser_commands.py` | Cleans subject transcripts and generates Stanford Parser shell commands. |
| `get_rule_counts.py` | Extracts syntactic rules, word tokens, POS entropy, and dependency-based measures. |
| `generate_syntac_subject_score.py` | Computes subject-level syntax scores from the extracted rules and exports `syntac_subject_score.csv`. |
| `SLT.R` | Assigns reference-corpus word/syntax frequencies and runs statistical analyses and plots. |

## Requirements

- **Python 3:** `numpy`, `pandas`, `nltk` (including NLTK WordNet data).
- **R:** `tidyverse`, `lme4`, `lmerTest`, `readxl`, `Hmisc`, `caret`, `nnet`.
- **Java 8+ and Stanford CoreNLP 3.9.2 (2018-10-05):** [Download the original version from Stanford](https://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip). The scripts invoke its `LexicalizedParser` and English PCFG model.

Install the Python dependencies and NLTK dictionary with:

```bash
pip install numpy pandas nltk
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Usage

Run from the module directory. Input transcripts, subject metadata, parsed files, and Switchboard reference frequency tables are **not distributed** with this repository.

1. Place transcripts in `Data/clean_data_raw_txt/`. If preparing Switchboard reference transcripts, provide the inputs expected by `process_switchboard.py` and run it first.
2. Generate cleaned transcripts and Stanford Parser commands:

   ```bash
   python write_parser_commands.py
   ```

3. Run the generated parser scripts from within `Data/` (e.g., `cd Data && bash run_parses0.sh`; repeat for the other batches).
4. Extract linguistic rules, then calculate subject-level syntax scores:

   ```bash
   python get_rule_counts.py
   python generate_syntac_subject_score.py
   ```

   Run `Rscript SLT.R` separately for the original syntax–lexicon analyses.

`SLT.R` additionally expects `Data/Study1Info.csv`, `Data/Study2Info.csv`, and the reference tables `Data/SwitchboardDicts/switchboard_token_large.csv` and `switchboard_syntax_large.csv`. These are not included. The R script contains the original commented code for constructing the reference frequency tables from Switchboard counts.

`generate_syntac_subject_score.py` reads `Data/syntaxrules.csv` and writes `../features/syntac_subject_score.csv` by default (relative to this module directory). Set `FHS_SYNTAX_RULES_FILE` and `FHS_FEATURE_DIR` to override its input file and output directory. It uses the original calculation: log of each rule count, averaged first within sentences and then across sentences for each subject. The output is read by the main feature-generation scripts.

The Python feature extractor writes `syntaxrules.csv`, `wordsrules_content.csv`, `wordsrules_nocontent.csv`, `alternative_metrics.csv`, and `pos_entropy.csv` under `Data/`. The R script carries out the downstream analyses and generates summary tables.

Synthetic example inputs may be provided separately; no real participant transcripts or metadata are included here.

## Original work and external resources

- **Original analysis and code:** Rezaii, N., Mahowald, K., Ryskin, R., Dickerson, B., & Gibson, E. (2022). *A syntax–lexicon trade-off in language production*. **PNAS**, 119(25), e2120203119. [Paper](https://doi.org/10.1073/pnas.2120203119) · [Original OSF project](https://osf.io/sr3ag/).
- **Stanford Parser:** Klein, D., & Manning, C. D. (2003). *Accurate Unlexicalized Parsing*. ACL. [Stanford Parser documentation](https://nlp.stanford.edu/software/lex-parser.html). See also [Manning et al. (2014), Stanford CoreNLP](https://aclanthology.org/P14-5010/).
- **Switchboard corpus:** Godfrey, J. J., Holliman, E. C., & McDaniel, J. (1992). *SWITCHBOARD: Telephone speech corpus for research and development*. ICASSP. [DOI](https://doi.org/10.1109/ICASSP.1992.225858) · [LDC corpus catalog](https://catalog.ldc.upenn.edu/LDC97S62).
