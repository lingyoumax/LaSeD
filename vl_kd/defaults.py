"""Default paths and model identifiers shared by CLI and training entrypoint."""

import os

# Hugging Face / ModelScope cache for downloaded weights
cache_dir = "./pretrainedmodels"

# Default backbone for both teacher and student unless overridden by CLI
MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"

# Cholec80-style layout: video folders under train/test, CSVs with frame labels
DATA_ROOT = "./data/cholec80"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
CSV_DIR = os.path.join(DATA_ROOT, "csvs")
