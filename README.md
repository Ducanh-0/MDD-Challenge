# Mispronunciation Detection and Diagnosis (MDD) using MTL + FA Architecture

This repository contains the official implementation of a Vietnamese Mispronunciation Detection and Diagnosis (MDD) system. The project is developed as a final assignment for the Speech Technology course. 

The core architecture utilizes a **Multi-Task Learning (MTL)** framework enhanced by a **Fused Cross-Attention (FA)** mechanism, built on top of a frozen **Wav2Vec2** acoustic backbone and a sequential **Bi-LSTM** diagnostic head.

---

## Architecture Overview & Data Flow

The model processes acoustic signals and structural texts simultaneously through two parallel tasks: Phoneme Recognition (ASR Task) and Mispronunciation Diagnosis (Diagnosis Task). The **Fused Cross-Attention** block acts as an information bridge, embedding soft phoneme probability distributions into the raw physical acoustic features.



1. **Acoustic Backbone:** `Wav2Vec2Model` extracts raw waveform representations.
2. **ASR Head:** Computes phonetic sequence distributions using CTC Loss.
3. **Fused Cross-Attention:** Uses ASR features as *Queries* and acoustic features as *Keys/Values* to align physical sound distortions with intended targets.
4. **Diagnosis Head:** A bidirectional LSTM processes the fused sequence contexts to classify each token into fine-grained evaluation states via Cross-Entropy Loss.

---

## Data Specifications & Formats

To ensure the pipelines execute correctly, your data files must strictly adhere to the following directory structure and formatting rules:

### 1. Training & Validation Datasets
The ground-truth annotation files must be saved in `.csv` format containing exactly **4 columns**: `id,path,canonical,transcript`
* `canonical`: The standard phonetic/phoneme string of the correct pronunciation.
* `transcript`: The actual phonetic/phoneme string pronounced by the speaker (human-labeled ground truth).

**Example (`train_data.csv` / `val_data.csv`):**
```csv
id,path,canonical,transcript
bia-dat_1618490611607,audio_data/public_test/bia-dat_1618490611607.wav,ɓ iə-5 ɗ a-0 tz,ɓ iə-0 ɗ a-0 tz
```
### 2. Inference / Test Dataset
The evaluation file used during inference contains exactly 3 columns: id,path,canonical (The transcript column is omitted).

**Example (`test_data.csv`):**
```csv
id,path,canonical
bia-dat_1618490611607,audio_data/public_test/bia-dat_1618490611607.wav,ɓ iə-5 ɗ a-0 tz
```
### 3. Phoneme Dictionary (lexicon.txt)
A plain text file containing the exhaustive list of unique phoneme tokens utilized across the dataset. This file is parsed at runtime to automatically construct the internal vocabulary mapping matrices (char2idx and idx2char).

**Example (`lexicon_vmd.txt`):**
```txt
1 m o-5 tz
a ə-0 iz
ai aː-0 iz
```
 Execution Platform & Path Configuration
Platform Requirements
The notebooks are optimized to run on Kaggle utilizing a GPU T4 x2 or GPU P100 environment.

## Path Customization
Before executing the pipeline, you must modify the path variables in the configuration cell (Cell 2 and Cell 8 - v9) within all three notebooks to point to your specific Kaggle dataset input directories:

```Python
class Config:
    TRAIN_CSV = "/kaggle/input/datasets/trngquangthi4139/mdd-challenge/MDD-Challenge-2025-training-set/metadata/train_phones.csv" 
    TRAIN_AUDIO_DIR = "/kaggle/input/datasets/trngquangthi4139/mdd-challenge/MDD-Challenge-2025-training-set/audio_data/train"

    PUBLIC_CSV = "/kaggle/input/datasets/trngquangthi4139/mdd-public-test-36/MDD-Challenge-2025-public-test/metadata/public_test_phones.csv"
    PUBLIC_AUDIO_DIR = "/kaggle/input/datasets/trngquangthi4139/mdd-public-test-36/MDD-Challenge-2025-public-test/audio_data/public_test"
    
    MODEL_CHECKPOINT = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
    BATCH_SIZE = 8
    EPOCHS = 60             
    LEARNING_RATE = 3e-5
    MAX_DURATION = 12.0
    DIAG_LOSS_TYPE = 'weighted_ce'

# inference
class PrivateTestConfig: 
    PRIVATE_CSV = "/kaggle/input/datasets/duckanh0/mdd-private-test/MDD-Challenge-2025-private-test/metadata/private_test_submission.csv"
    
    PRIVATE_AUDIO_DIR = "/kaggle/input/datasets/duckanh0/mdd-private-test/MDD-Challenge-2025-private-test/audio_data/private_test"
    
    MODEL_WEIGHTS = "/kaggle/working/best_mdd_model.pt"
```
## Sequential Multi-Stage Fine-Tuning Workflow
To replicate our peak performance on the competition leaderboard, the notebooks must be executed sequentially. This pipeline enforces a progressive optimization strategy:
```
[ Base Training ]      ──► mtl-fa-4-0.ipynb (Run Full Epochs)
                                 │
                                 ▼ (Exports best_mdd_model.pt)
[ Fine-Tuning Phase 1 ] ──► mtl-fa-4-1-v7.ipynb (Loads Phase 1 Weights -> Early Stopping at Epoch 3)
                                 │
                                 ▼ (Exports best_mdd_model.pt)
[ Fine-Tuning Phase 2 ] ──► mtl-fa-4-1-v9.ipynb (Loads Phase 2 Weights -> Runs completely to final submission)
```
### Step 1: Base Training (mtl-fa-4-0.ipynb)
Open the notebook on Kaggle, configure your input dataset paths, and trigger Run All.

The model establishes the initial shared representation spaces and cross-attention maps. The notebook automatically evaluates checkpoint states using a standard tri-sequence alignment algorithm and saves the best model to /kaggle/working/best_mdd_model.pt.

### Step 2: Incremental Fine-Tuning Phase 1 (mtl-fa-4-1-v7.ipynb)
Load the checkpoint file best_mdd_model.pt generated from Step 1 into this notebook as warm-start initial weights.

Run the training process. Empirical metrics show that convergence plateaus drastically beyond the 3rd epoch. Perform a Manual Early Stopping at Epoch 3 to protect the model from overfitting and save the peak checkpoint.

### Step 3: Global Fine-Tuning Phase 2 (mtl-fa-4-1-v9.ipynb)
Import the checkpoint file preserved from Step 2 (v7) into this final script.

Run the configuration to its final epoch. This stage refines decision boundaries and minimizes diagnostic target variances, outputting the most robust weights for test set inference.

## Inference & Output Generation
Upon completing the test inference block at the end of the mtl-fa-4-1-v9.ipynb notebook, the system automatically writes a submission file (submission_private.csv) to your /kaggle/working/ output directory.

The output complies with a structured 3-column format: id,path,predict

Example Output:
```csv
id,path,predict
vao-nui_1618754903173,audio_data/private_test/vao-nui_1618754903173.wav,v aː-1 uz n w i-4
The predict column stores the final model-diagnosed phoneme sequence output, indicating specific mispronunciation patterns ready for automated grading engines.
```
