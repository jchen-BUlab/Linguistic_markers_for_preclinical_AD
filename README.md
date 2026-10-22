# Speech-Based Linguistic Feature Analysis

We aimed to systematically evaluate a broad range of linguistically motivated features derived from delayed recall responses as potential markers of pre-clinical Alzheimer’s disease. We used natural language processing techniques to automate feature extraction and applied statistical and machine learning methods to examine associations between these features and global amyloid status and regional tau burden. Code developed for the Linguistic Markers for Preclinical AD project and the 2026 paper accepted by Journal of Prevention of Alzheimer's Disease.

Huang H, Li Z, Young CB, Smith V, Ang, TFA, Kolachalama, VB, Au R, Chen J. Identifying Linguistic Features from Constrained Spontaneous Speech as Potential Markers of Pre-clinical Alzheimer’s Disease. Accepted by The Journal of Prevention of Alzheimer's Disease.

copyright (c) 2026 Haochun Huang, Jinying Chen, Boston University Chobanian & Avedisian School of Medicine

Licensed under the MIT License (the "License"); you may not use this file except in compliance with the License.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## Repository structure

```text
env.yaml                                  # All python package used in this study with their version
R_env.txt                                 # All R package used in this study with their version
README.md
Example_Data/
    transcripts/SYN001.csv, SYN002.csv     # Fictional transcript examples
    syntaxrules_example.csv                 # Fictional syntax rules
    features/syntac_subject_score.csv       # Scores generated from fictional rules
Code/
├── data_util.py                         # Input/output paths shared by feature extractors
├── feature_generation.py                # Standard linguistic feature extraction
├── feature_generation_filter.py         # Linguistic features after utterance filtering (for sensitivity analysis)
├── Data_link_Final.Rmd                  # Link linguistic features with NP and PET measurements
├── speech_markers_amyloid.Rmd           # Regression analyses: standard linguistic features and amyloid / tau (primary analysis)
├── speech_markers_amyloid_filtered.Rmd  # Regression analyses: filtered features and amyloid / tau (sensitivity analysis)
├── feature_distribution.Rmd            # Feature histograms; regression analysis: lingustic features and LMd
├── boruta_feature_amyloid.Rmd          # Boruta selection for amyloid and tau outcomes
├── logistic_lasso.Rmd                   # LASSO analyses using features selected by individual regression analyses
├── Feature_Cluster.ipynb                # Feature correlations and hierarchical clustering
├── combined_plot.Rmd                    # Combined regression, LASSO, and Boruta plots
├── Sensitivity_analysis.Rmd            # Sensitivity analysis for continous Amyloid status
└── Syntax_frequency_SLT/
    ├── process_switchboard.py          # Prepare Switchboard transcripts
    ├── write_parser_commands.py        # Prepare Stanford Parser commands
    ├── get_rule_counts.py              # Extract syntax rules and related metrics
    ├── generate_syntac_subject_score.py # Calculate subject-level syntax scores
    ├── SLT.R                            # Original syntax–lexicon frequency analyses
    └── README.md                        # Module instructions and source citations
```

## Analysis workflow

1. **Syntax frequency:** Prepare transcripts and Stanford Parser output using `Syntax_frequency_SLT/`. Run `get_rule_counts.py`, then `generate_syntac_subject_score.py` to produce `syntac_subject_score.csv`. See the [syntax-frequency README](Code/Syntax_frequency_SLT/README.md) for details and references.
2. **Linguistic feature extraction:** Run `feature_generation.py` (normal) or `feature_generation_filter.py` (filtered). Both calculate lexical, part-of-speech, syntactic-complexity, speech-rate, pause, and related features. The filtered version additionally excludes selected noise-related utterances from specified calculations.
3. **Data linkage:** Run `Data_link_Final.Rmd` to match speech features to neuropsychological (NP) and amyloid/tau PET data. The same script handles both input variants.
4. **Regression analyses:** Run `speech_markers_amyloid.Rmd` or `speech_markers_amyloid_filtered.Rmd`, matching the selected variant. These scripts produce regression results, figures, and an RDS file for subsequent analyses.
5. **Feature characterization and selection:** Run `feature_distribution.Rmd` (distributions and LMd associations), `boruta_feature_amyloid.Rmd` (Boruta feature selection), and `Feature_Cluster.ipynb` (correlation-based feature clustering and leaf order).
6. **LASSO and combined visualization:** Run `logistic_lasso.Rmd` after regression and Boruta results are available. Then run `combined_plot.Rmd` after LASSO and feature-clustering outputs have been generated.
7. **Revised analyses (separate):** `Revise_Code.Rmd` uses a separately prepared revised input dataset and is not part of the automatic normal/filtered workflow.

## Main inputs and outputs

| Script / stage | Main input | Main output |
| --- | --- | --- |
| `Syntax_frequency_SLT/` | Subject/reference transcripts and Stanford Parser results | `syntaxrules.csv`, `syntac_subject_score.csv`, word-rule tables, POS/dependency metrics |
| `feature_generation.py` | Cleaned transcript CSVs; `syntac_subject_score.csv` | `linguistic_feature_spacy_clean_20251116.csv` |
| `feature_generation_filter.py` | Same inputs, with utterance filtering | `linguistic_feature_spacy_clean_filtered_20251116.csv` |
| `Data_link_Final.Rmd` | Matching feature CSV; NP and PET tables | `feature_np_tau_amyloid_20251116.csv` or `feature_np_tau_amyloid_filtered_20251116.csv` |
| `speech_markers_amyloid*.Rmd` | Corresponding linked feature CSV | `tauFeatureAmyloidDemo.rds` or `tauFeatureAmyloidDemo_filtered.rds`; regression tables/figures |
| `feature_distribution.Rmd` | Linked feature CSV; corresponding RDS | `hist_<feature>.png`, `Feature_LMd_regression_comparison.csv`, `feature_lmd_scatterplot_estimate.png` |
| `boruta_feature_amyloid.Rmd` | Corresponding RDS | `boruta_amyloidFeatureLMdDemo.csv`, `boruta_tauAmyloidFeatureLMdDemo.csv` |
| `logistic_lasso.Rmd` | Corresponding RDS; regression-selection and Boruta CSVs | `lasso_lm_tau.csv`, `lasso_boruta_tau.csv`, `lasso_boruta_amyloid.csv` |
| `Feature_Cluster.ipynb` | Linked feature CSV | `Linguistic_feature_dendrogram_cluster.csv`, `Linguistic_feature_dendrogram_leaf_label.csv`, tree/dendrogram files |
| `combined_plot.Rmd` | Regression, Boruta and LASSO tables; dendrogram leaf-order CSV | `combined_plot_updated.png`, `combined_plot_updated.tif` |
| `Revise_Code.Rmd` | `feature_np_tau_amyloid_20260512_revise.csv` | Additional revised regression tables |

**Data availability:** No real participant transcripts, subject metadata, NP/PET measurements, or restricted reference-corpus data are included. The datasets analyzed in this study are available to qualified investigators through a formal research application to the Fram-ingham Heart Study (https://www.framinghamheartstudy.org/fhs-for-
researchers/). Access may be subject to institutional and data-use approvals. Generated `selected_noise_segments_*.csv` files may contain transcript excerpts and should not be made public.

Default output locations are `Results/Data/`, `Results/Table_Results/`, and `Results/Figure_Results/`. The filtered statistical workflow uses `Results/Table_Results_filtered/` and `Results/Figure_Results_filtered/`; feature histograms use their own distribution subdirectories.

## Synthetic input examples

The `Example_Data/` folder contains **entirely fictional** data illustrating the expected input formats. These files are for demonstrating column names and file organization, **not** for reproducing study results.

**Transcript CSV** — one file per recording/subject (for example, [`Example_Data/transcripts/SYN001.csv`](Example_Data/transcripts/SYN001.csv)). The original transcript format uses these columns:

```csv
cohort,ID,date,tier,name,start_time,end_time,duration,contents,task,clean_contents,clean_contents_w_selfcor,segment_id,comment,selfcorrection
SYNTHETIC,SYN001,2026-01-01,INV,Interviewer,0,1500,1500,What is happening in this picture?,PictureDescription,What is happening in this picture?,What is happening in this picture?,SYN001_001,,0
SYNTHETIC,SYN001,2026-01-01,PAR,Participant,1800,4000,2200,The family is having a picnic.,PictureDescription,The family is having a picnic.,The family is having a picnic.,SYN001_002,,0
```

Here, `PAR` identifies participant speech and `INV` interviewer speech. `start_time`, `end_time`, and `duration` are in milliseconds. The feature-generation scripts specifically use `ID`, `tier`, `contents`, `duration`, `start_time`, and `end_time`. Each filename stem must match the `subjectID` in the syntax-score table used by the feature extractor.

**Syntax-rule input** — [`Example_Data/syntaxrules_example.csv`](Example_Data/syntaxrules_example.csv) illustrates the output format of `get_rule_counts.py` used by `generate_syntac_subject_score.py`:

```csv
SubjectID,SentenceID,SyntaxRule_original
SYN001,SYN001_Sentence1,nsubj + VERB
SYN001,SYN001_Sentence1,det + NOUN
```

**Subject-level syntax scores** — [`Example_Data/features/syntac_subject_score.csv`](Example_Data/features/syntac_subject_score.csv) is generated from the synthetic syntax rules. The relevant columns are `subjectID` and `syntac_score`, which are merged into linguistic features by the Python scripts. The generated example retains the original CSV index column. To regenerate this example from the repository root:

```bash
FHS_SYNTAX_RULES_FILE=Example_Data/syntaxrules_example.csv \
FHS_FEATURE_DIR=Example_Data/features \
python Code/Syntax_frequency_SLT/generate_syntac_subject_score.py
```

Clinical, NP/PET, and full reference-corpus inputs are **not** supplied as example study datasets.

## Running the code

Run the main workflow from the `Code/` directory. The synthetic examples illustrate structure but do not replace the restricted inputs required for the complete analysis. Python dependencies include `pandas`, `numpy`, `nltk`, `spaCy`, and `benepar` with the `en_core_web_sm` and `benepar_en3` models. Feature clustering also uses scientific Python plotting/clustering packages. R scripts require the packages imported in each file (including `rmarkdown` for rendering). The syntax-frequency module additionally requires the Stanford Parser; installation and citations are in its README.

Set paths to the private input data and results folders as needed:

```bash
export FHS_DATA_DIR=/path/to/private/data
export FHS_TRANSCRIPTS_DIR=/path/to/cleaned/transcripts
export FHS_FEATURE_DIR=/path/to/features
export FHS_RESULTS_DIR=/path/to/Results
export FHS_OUTPUT_DIR=/path/to/Results/Data
# Optional: FHS_FEATURE_INPUT_DIR=/path/to/feature/CSVs
```

Cleaned transcript CSVs should contain `ID`, `tier`, `contents`, `duration`, `start_time`, and `end_time`. The subject-level syntax-score file is produced by `Syntax_frequency_SLT/generate_syntac_subject_score.py` and placed in `FHS_FEATURE_DIR`.

**Normal features:**

```bash
export FHS_FEATURE_VARIANT=normal
python feature_generation.py
Rscript -e "rmarkdown::render('Data_link_Final.Rmd')"
Rscript -e "rmarkdown::render('speech_markers_amyloid.Rmd')"
```

**Filtered features:**

```bash
export FHS_FEATURE_VARIANT=filtered
python feature_generation_filter.py
Rscript -e "rmarkdown::render('Data_link_Final.Rmd')"
Rscript -e "rmarkdown::render('speech_markers_amyloid_filtered.Rmd')"
```

For either variant, continue with downstream analyses (keeping the same `FHS_FEATURE_VARIANT` value):

```bash
Rscript -e "rmarkdown::render('feature_distribution.Rmd')"
Rscript -e "rmarkdown::render('boruta_feature_amyloid.Rmd')"
# Run Feature_Cluster.ipynb in Jupyter
Rscript -e "rmarkdown::render('logistic_lasso.Rmd')"
Rscript -e "rmarkdown::render('combined_plot.Rmd')"
```

`combined_plot.Rmd` reads `Linguistic_feature_dendrogram_leaf_label.csv` from `Results/Table_Results/` by default. To use the filtered feature ordering, set `FHS_CLUSTER_DIR` to `Results/Table_Results_filtered/` (or its absolute path). `Revise_Code.Rmd` is run separately; its input can be set with `FHS_REVISED_INPUT_FILE`.

Some scripts intentionally retain fixed row/column selections from the original analyses. Check the input table structure and order when reproducing the results on other datasets.
