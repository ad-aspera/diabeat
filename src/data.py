import os
import pickle
import random
from typing import Optional, Tuple, Literal
import numpy as np
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as pl


SLEEP_STAGES = ["DS", "REM", "RS"]


@dataclass
class HRVDataConfig:
    # path to the hrv data directory
    hrv_data_dir: str
    # split to use
    split: Literal["train", "val"]
    # number of peaks per sample
    # 600 peaks ~= 10 minutes of recording
    n_peaks_per_sample: int = 600

    # data loader config params
    train_batch_size: int = 8
    val_batch_size: int = 8

    train_num_workers: int = 4
    val_num_workers: int = 4

    train_shuffle: bool = True
    val_shuffle: bool = False

    train_pin_memory: bool = True
    val_pin_memory: bool = True


class HRVDataset(Dataset):
    def __init__(self, config: HRVDataConfig):
        """
        Args:
            hrv_data_dir (str): Directory containing ECG CSV files
            split (str): "train" or "val"
            participant_characteristics_path (str): Path to participant characteristics CSV
        """
        # Load raw HRV data
        hrv_data_path = os.path.join(config.hrv_data_dir, f"{config.split}.pkl")
        with open(hrv_data_path, "rb") as f:
            self.raw_data = pickle.load(f)

        self.n_peaks_per_sample = config.n_peaks_per_sample

    def __len__(self) -> int:
        return len(self.raw_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # return a random chunk of data from a random sleep stage
        sleep_stage = random.choice(SLEEP_STAGES)

        features = self.raw_data[idx]
        hrv_data = features[sleep_stage][0]

        # obtain a random start index of the chunk
        start_idx = random.randint(0, len(hrv_data) - self.n_peaks_per_sample + 1)
        end_idx = start_idx + self.n_peaks_per_sample

        x = hrv_data[start_idx:end_idx].astype(np.float32)
        label = features["diabetic_peripheral_neuropathy"].astype(np.float32)
        # return the chunk of data and the features
        return (x, label)


class HRVDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: HRVDataConfig,
    ):
        super().__init__()
        self.config = config

    def setup(self, stage: Optional[str] = "fit"):
        if stage == "fit":
            self.train_dataset = HRVDataset(self.config)
            self.val_dataset = HRVDataset(self.config)
        # val dataset is used for testing cause we don't have much data
        elif stage == "test":
            self.test_dataset = HRVDataset(self.config)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=self.config.train_shuffle,
            num_workers=self.config.train_num_workers,
            pin_memory=self.config.train_pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.val_batch_size,
            shuffle=self.config.val_shuffle,
            num_workers=self.config.val_num_workers,
            pin_memory=self.config.val_pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.val_batch_size,
            shuffle=self.config.val_shuffle,
            num_workers=self.config.val_num_workers,
            pin_memory=self.config.val_pin_memory,
        )
