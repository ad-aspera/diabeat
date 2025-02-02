import os
import pickle
import random
from typing import Optional, Tuple, Literal
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as pl


SLEEP_STAGES = ["DS", "REM", "RS"]
dpn_label = "diabetic_peripheral_neuropathy"


class HRVDataset(Dataset):
    def __init__(self, config: OmegaConf):
        """
        Args:
            config (OmegaConf): Omegaconf configuration object containing data parameters
        """
        # Load raw HRV data
        hrv_data_path = os.path.join(config.hrv_data_dir, f"{config.split}.pkl")
        with open(hrv_data_path, "rb") as f:
            self.raw_data = pickle.load(f)

        self.n_peaks_per_sample = config.n_peaks_per_sample

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.raw_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - x: HRV signal tensor of shape (n_peaks_per_sample,)
                - label: Binary label tensor indicating presence of diabetic peripheral neuropathy
        """
        # return a random chunk of data from a random sleep stage
        sleep_stage = random.choice(SLEEP_STAGES)

        features = self.raw_data[idx]
        hrv_data = features[sleep_stage][0]

        # Check if we have enough data points
        if len(hrv_data) < self.n_peaks_per_sample:
            print("FOUND SHORT DATA", features)
            # Either pad with zeros/repeat the data
            hrv_data = np.pad(
                hrv_data, (0, self.n_peaks_per_sample - len(hrv_data)), mode="wrap"
            )

        # obtain a random start index of the chunk
        start_idx = random.randint(0, len(hrv_data) - self.n_peaks_per_sample)
        end_idx = start_idx + self.n_peaks_per_sample

        x = hrv_data[start_idx:end_idx].astype(np.float32)
        label = features[dpn_label].astype(np.float32)

        # Convert to tensor here to ensure consistent shape
        x = torch.from_numpy(x)
        label = torch.tensor(label)

        return (x, label)


class HRVDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: OmegaConf,
    ):
        """Initialize the HRV data module.

        Args:
            config (OmegaConf): Configuration object containing data parameters
        """
        super().__init__()
        self.config = config

    def setup(self, stage: Optional[str] = "fit"):
        """Set up train, validation and test datasets.

        Args:
            stage (Optional[str]): Pipeline stage - either 'fit' or 'test'
        """
        if stage == "fit":
            self.train_dataset = HRVDataset(self.config)
            self.val_dataset = HRVDataset(self.config)
        # val dataset is used for testing cause we don't have much data
        elif stage == "test":
            self.test_dataset = HRVDataset(self.config)

    def train_dataloader(self):
        """Create the training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=self.config.train.shuffle,
            num_workers=self.config.train.num_workers,
            pin_memory=self.config.train.pin_memory,
        )

    def val_dataloader(self):
        """Create the validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.val.batch_size,
            shuffle=self.config.val.shuffle,
            num_workers=self.config.val.num_workers,
            pin_memory=self.config.val.pin_memory,
        )

    def test_dataloader(self):
        """Create the test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.val.batch_size,
            shuffle=self.config.val.shuffle,
            num_workers=self.config.val.num_workers,
            pin_memory=self.config.val.pin_memory,
        )
