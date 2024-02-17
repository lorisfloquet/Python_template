#!/usr/bin/env python3

import torch
from termcolor import colored

import os


# The root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# The directory containing the source code
SRC_DIR = os.path.join(ROOT_DIR, "src")

# The directory containing the data
DATA_DIR = os.path.join(ROOT_DIR, "data")

# The directory containing the datasets
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")

# The directory to the base output directory
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

# Determine the current device and based on that set the pin memory flag
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False
NB_WORKERS = os.cpu_count() if DEVICE == "cuda" else 0
print(colored(f"**** using {DEVICE.upper()}", "light_green"))


########################################################################################################################
# Parameters for the model
########################################################################################################################

# Whether to load the model from a checkpoint
LOAD_MODE = False

# The dataset parameters
NB_TRAIN_INSTANCES = 100_000
NB_VAL_INSTANCES = 10_000
NB_TEST_INSTANCES = 10_000
TRAIN_DATASET_PATH = os.path.join(DATASETS_DIR, "train_dataset.txt")
TEST_DATASET_PATH = os.path.join(DATASETS_DIR, "test_dataset.txt")
VAL_DATASET_PATH = os.path.join(DATASETS_DIR, "val_dataset.txt")

# Initialize our initial learning rate, number of epochs to train for and the batch size
INIT_LR = 1e-4
NUM_EPOCHS = 10
NUM_EPOCHS_DONE = 0
BATCH_SIZE = 64

# Define paths to output model, plot and testing paths for WavSpa
ATTEMPT_NB = 0
SUFFIX = f"{NUM_EPOCHS}_{BATCH_SIZE}_{ATTEMPT_NB}"
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, f"best_model_{SUFFIX}.pth")
LAST_MODEL_PATH = os.path.join(OUTPUT_DIR, f"last_model_{SUFFIX}.pth")
TEST_PATH = os.path.join(OUTPUT_DIR, f"test_data_{SUFFIX}.csv")
VAL_PATH = os.path.join(OUTPUT_DIR, f"val_data_{SUFFIX}.csv")
TRAIN_PATH = os.path.join(OUTPUT_DIR, f"training_data_{SUFFIX}.csv")
PLOT_DATA_PATH = os.path.join(OUTPUT_DIR, f"plot_data_{SUFFIX}.csv")
PLOT_PATH = os.path.join(OUTPUT_DIR, f"convergence_plot_{SUFFIX}.png")
