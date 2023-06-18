import glob
import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from tqdm.notebook import tqdm


def resample_seq_df(df, in_hz, out_hz, with_classes=True, with_bool_cols=True):
    in_ms = (1 / in_hz) * 1000
    out_ms = (1 / out_hz) * 1000
    FLOAT_COLS = ["AccV", "AccML", "AccAP"]
    if with_classes:
        CLASSES_COLS = ["StartHesitation", "Turn", "Walking"]
    if with_bool_cols:
        BOOL_COLS = ["Valid", "Task"]

    df["Time"] = pd.to_timedelta(df["Time"] * in_ms, unit="ms")
    df = df.set_index("Time")

    resampled_df = (
        df[FLOAT_COLS]
        .resample(f"{out_ms}ms")
        .mean()  # new val = "mean" in the 7.8125ms interval
        .interpolate()  # sometimes there is no previous value in the 7.8125ms
        # interval: we interpolate (linearly by default)
    )

    cols = []
    if with_classes:
        cols = cols + CLASSES_COLS
    if with_bool_cols:
        cols = cols + BOOL_COLS
    if cols != []:
        resampled_df[cols] = (
            df[cols]
            .resample(f"{out_ms}ms")
            .first()
            .ffill()  # new val = previous val
        )

    # needed as the introduction of NaNs forced pd to make all cols float
    if with_classes:
        resampled_df[CLASSES_COLS] = resampled_df[CLASSES_COLS].astype(int)
    if with_bool_cols:
        resampled_df[BOOL_COLS] = resampled_df[BOOL_COLS].astype(bool)

    return resampled_df


def convert_g_to_ms2(df):
    # 1g = 9.80665m/s^2
    df["AccV"] = df["AccV"] * 9.80665
    df["AccML"] = df["AccML"] * 9.80665
    df["AccAP"] = df["AccAP"] * 9.80665
    return df


def add_noactivity_lbls(seq_lbls):
    noactivity_col = np.zeros((seq_lbls.shape[0], 1))
    noactivity_indices = np.sum(seq_lbls, axis=1) == 0.0
    noactivity_col[noactivity_indices] = 1.0
    seq_lbls = np.hstack((seq_lbls, noactivity_col))
    return seq_lbls


def normalize(seq_features):
    seq_features = (
        seq_features - seq_features.mean(axis=0)
    ) / seq_features.std(axis=0)
    return seq_features


def downsample_seq(seq_inhz, in_hz, out_hz):
    out_size = int(seq_inhz.shape[0] * (out_hz / in_hz))
    time_inhz = np.linspace(0, 1, seq_inhz.shape[0])
    time_outhz = np.linspace(0, 1, out_size)

    seq_outhz = np.zeros((out_size, seq_inhz.shape[1]))

    for i in range(seq_inhz.shape[1]):
        interp_func = interp1d(time_inhz, seq_inhz[:, i])
        seq_outhz[:, i] = interp_func(time_outhz)

    return seq_outhz


def preprocess_defog(defog_folder_path, ds_part=1.0, down_hz=None):
    paths = glob.glob(os.path.join(defog_folder_path, "**"))
    paths = paths[: round(len(paths) * ds_part)]
    all_features = []
    all_lbls = []
    all_masks = []
    for i in tqdm(range(len(paths))):
        seq = pd.read_csv(paths[i])

        if down_hz is None:
            # upsampling the data from 100Hz to 128Hz
            seq = resample_seq_df(seq, 100, 128)
        else:
            # downsample the data from 100Hz to ??Hz
            seq = resample_seq_df(seq, 100, down_hz)

        # defog data is in g: we convert it into m/s^2
        seq = convert_g_to_ms2(seq)

        # get the associated mask
        seq_mask = (seq["Valid"] & seq["Task"]).values

        # extracting the features and the labels separately
        seq_features = seq[["AccV", "AccML", "AccAP"]].values
        seq_lbls = seq[["StartHesitation", "Turn", "Walking"]].values

        # add a 4th label associated with no activity
        # ie which is 1 when no other class is active
        seq_lbls = add_noactivity_lbls(seq_lbls)

        all_features.append(seq_features)
        all_lbls.append(seq_lbls)
        all_masks.append(seq_mask)

    return all_features, all_lbls, all_masks


def preprocess_tdcsfog(tdcsdefog_folder_path, ds_part=1.0, down_hz=None):
    paths = glob.glob(os.path.join(tdcsdefog_folder_path, "**"))
    paths = paths[: round(len(paths) * ds_part)]
    all_features = []
    all_lbls = []
    all_masks = []
    for i in tqdm(range(len(paths))):
        seq = pd.read_csv(paths[i])

        if down_hz is not None:
            # downsample the data from 128Hz to ??Hz
            seq = resample_seq_df(
                seq, 128, down_hz, with_classes=True, with_bool_cols=False
            )

        # get the associated mask: for tdcsfog there are no Valid / Task labels,
        # so we consider the whole sequence
        seq_mask = np.array([True for _ in range(seq.shape[0])])

        # extracting the features and the labels separately
        seq_features = seq[["AccV", "AccML", "AccAP"]].values
        seq_lbls = seq[["StartHesitation", "Turn", "Walking"]].values

        # add a 4th label associated with no activity
        # ie which is 1 when no other class is active
        seq_lbls = add_noactivity_lbls(seq_lbls)

        all_features.append(seq_features)
        all_lbls.append(seq_lbls)
        all_masks.append(seq_mask)

    return all_features, all_lbls, all_masks


def normalize(X_train):
    # per seq norm
    return [StandardScaler().fit_transform(seq) for seq in X_train]


class SeqFoGDataset(Dataset):
    def __init__(
        self,
        defog_root,
        tdcs_root,
        ds_part=1.0,
        max_seq_len=None,
        down_hz=None,
    ):
        self.defog_root = defog_root
        self.tdcs_root = tdcs_root

        defog = preprocess_defog(defog_root, ds_part=ds_part, down_hz=down_hz)
        tdcs = preprocess_tdcsfog(tdcs_root, ds_part=ds_part, down_hz=down_hz)

        X = normalize(defog[0] + tdcs[0])

        self.X = [torch.from_numpy(xi).float()[:max_seq_len] for xi in X]
        self.y = [
            torch.from_numpy(yi).float()[:max_seq_len]
            for yi in defog[1] + tdcs[1]
        ]
        self.masks = [
            torch.from_numpy(m).bool()[:max_seq_len] for m in defog[2] + tdcs[2]
        ]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.masks[idx]

