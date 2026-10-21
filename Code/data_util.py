"""Input/output locations for the linguistic feature extraction pipeline.

Paths are relative to the working directory by default. Set the corresponding
FHS_* environment variables to use external, non-public data locations.
No speech transcripts or subject metadata are distributed with these scripts.
"""

import os

# Directory containing project input data (not included in the repository).
data_dir = os.environ.get("FHS_DATA_DIR", "Data")

# Cleaned transcript CSVs with ID, tier, contents, duration and timing columns.
data_voice_clean_dir = os.environ.get(
    "FHS_TRANSCRIPTS_DIR", os.path.join(data_dir, "dvoice_transcripts_clean")
)

# Directory containing the previously computed syntax-frequency scores.
feat_dir = os.environ.get("FHS_FEATURE_DIR", "features")

# Local directory for generated subject-level features and noise-segment logs.
output_dir = os.environ.get("FHS_OUTPUT_DIR", os.path.join("Results", "Data"))
