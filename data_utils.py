import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from random import randrange
import random
import json
import pandas
import eval_metrics as em

def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []

    # Read the JSON file
    with open(dir_meta, 'r') as f:
        data = json.load(f)

    if is_train:
        for item in data:
            key = item['file_path']
            label = item['label']
            file_list.append(key)
            d_meta[key] = 1 if label == 'real' else 0
        return d_meta, file_list

    elif is_eval:
        for item in data:
            key = item['file_path']
            file_list.append(key)
        return file_list

    else:
        for item in data:
            key = item['file_path']
            label = item['label']
            file_list.append(key)
            d_meta[key] = 1 if label == 'real' else 0
        return d_meta, file_list



def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	
			

class ADD_Dataset(Dataset):
    def __init__(self, args, list_IDs, labels, is_eval=False):
        """
        Args:
            args: Additional arguments
            list_IDs: List of file paths
            labels: Dictionary mapping file paths to labels
        """
        self.list_IDs = list_IDs
        self.labels = labels
        self.args = args
        self.cut = 64600  # ~4 seconds of audio (64600 samples)
        self.is_eval = is_eval
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        filepath = self.list_IDs[index]
        # filepath = os.path.join(self.base_dir, filepath)
        X, fs = librosa.load(filepath, sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        target = self.labels[filepath]
        if self.is_eval:
            return x_inp, filepath
        return x_inp, target



def eval_to_score_file(score_file, key_json_file):
    """
    Evaluate scores and calculate EER based on ground truth JSON file.
    """
    # Load ground truth from JSON file
    with open(key_json_file, 'r') as f:
        cm_data = json.load(f)

    # Load the submission scores
    submission_scores = pandas.read_csv(score_file, sep='|', header=None, skipinitialspace=True)

    if len(submission_scores.columns) != 2:
        print("Error: Submission file must have exactly 2 columns (file_name and score).")
        exit(1)

    # Convert cm_data JSON to a DataFrame
    cm_df = pandas.DataFrame(cm_data)

    # Merge scores with ground truth
    cm_scores = submission_scores.merge(cm_df, left_on=0, right_on="file_path", how="inner")

    # Extract bona-fide and spoof scores
    bona_cm = cm_scores[cm_scores["label"] == "real"][1].values
    spoof_cm = cm_scores[cm_scores["label"] == "fake"][1].values

    # Compute EER
    eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]
    print(f"EER: {eer_cm * 100:.2f}%")
    return eer_cm
