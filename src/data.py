import os
import pickle
import random
from typing import List, Dict, Tuple, Union

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader


class HRVDataset(Dataset):
    def __init__(self, config: OmegaConf, fold: int = 0, split: str = "train"):
        """
        Args:
            config (OmegaConf): Omegaconf configuration object containing data parameters
        """
        # Load raw HRV data
        hrv_data_path = os.path.join(
            config.hrv_data_dir,
            config.data_config,
            f"fold_{fold}",
            f"{split}.pkl",
        )
        with open(hrv_data_path, "rb") as f:
            self.raw_data = pickle.load(f)

        self.n_peaks_per_sample = config.n_peaks_per_sample

        # Calculate min and max values across all data for scaling
        self.min_val = config.min_hrv_threshold
        self.max_val = config.max_hrv_threshold

        self.slice_strategy = config.slice_strategy
        self.class_config = config.class_config

        self.clean_data()
        self.merge_classes()
        self.chunks = self.generate_chunks()

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Get a single sample from the dataset."""
        return (
            torch.tensor(self.chunks[idx]["data"], dtype=torch.float32),
            torch.tensor(self.chunks[idx]["label"], dtype=torch.long),
            self.chunks[idx]["patient_id"],
        )

    def clean_data(self) -> None:
        """Clean the data by removing samples with HRV values outside the threshold."""
        # Create a new list to store cleaned data
        cleaned_data = []
        total_values = 0
        outliers = 0

        # Iterate through each patient's data
        for patient in self.raw_data:
            # Get the HRV values
            hrv_values = patient["data"]
            total_values += len(hrv_values)

            # Filter out values outside threshold range
            valid_indices = np.logical_and(
                hrv_values >= self.min_val, hrv_values <= self.max_val
            )
            outliers += np.sum(~valid_indices)

            # Only keep patient if they have valid data after filtering
            if np.any(valid_indices):
                patient["data"] = hrv_values[valid_indices]
                cleaned_data.append(patient)

        # Update raw_data with cleaned version
        self.raw_data = cleaned_data

        print(
            f"Removed {outliers} outlier values out of {total_values} total values ({(outliers/total_values)*100:.2f}%)"
        )

    def merge_classes(self) -> None:
        """Merge classes based on class_config."""
        assert self.class_config in ["all", "diab_v_comp", "comp_v_dpn"]
        if self.class_config == "all":
            return
        elif self.class_config == "diab_v_comp":
            print("Merging complications and DPN into class 1")
            # Merge complications (1) and DPN (2) into class 1
            for patient in self.raw_data:
                if patient["class"] == 2:
                    patient["class"] = 1
        elif self.class_config == "comp_v_dpn":
            print(
                "Merging diabetes and complications into class 0 and DPN into class 1"
            )
            # Merge diabetes (0) and complications (1) into 0, and DPN (2) -> 1
            for patient in self.raw_data:
                if patient["class"] == 2:
                    patient["class"] = 1
                elif patient["class"] == 1:
                    patient["class"] = 0

    def generate_min_max_scale_values_from_train_set(
        self, hrv_data_dir: str, data_config: str
    ) -> Tuple[float, float]:
        """Generate min and max values across all data for scaling."""
        # use train and fold 0 even for scaling the validation set
        with open(
            os.path.join(
                hrv_data_dir,
                data_config,
                "fold_0/train.pkl",
            ),
            "rb",
        ) as f:
            train_set = pickle.load(f)
        all_data = np.concatenate([sample["data"] for sample in train_set])
        return np.min(all_data), np.max(all_data)

    def generate_chunks(self) -> List[Dict[str, Union[np.ndarray, int, str]]]:
        """Generate chunks of data based on slice strategy.

        Returns:
            List[Dict]: List of dictionaries containing:
                - data: Min-max scaled HRV signal array
                - label: Class label (0, 1, or 2)
                - patient_id: ID of the patient
        """
        assert self.slice_strategy in ["sliding", "chunked"]
        chunks = []

        for patient in self.raw_data:
            hrv_data = self.min_max_scale(patient["data"])
            label = patient["class"]
            patient_id = patient["id"]

            if len(hrv_data) < self.n_peaks_per_sample:
                # Pad if too short
                hrv_data = np.pad(
                    hrv_data, (0, self.n_peaks_per_sample - len(hrv_data)), mode="wrap"
                )

            if self.slice_strategy == "sliding":
                # Sliding window with stride=1
                for start_idx in range(len(hrv_data) - self.n_peaks_per_sample + 1):
                    chunk = hrv_data[start_idx : start_idx + self.n_peaks_per_sample]
                    chunks.append(
                        {"data": chunk, "label": label, "patient_id": patient_id}
                    )

            elif self.slice_strategy == "chunked":
                # Non-overlapping chunks
                for start_idx in range(
                    0,
                    len(hrv_data) - self.n_peaks_per_sample + 1,
                    self.n_peaks_per_sample,
                ):
                    chunk = hrv_data[start_idx : start_idx + self.n_peaks_per_sample]
                    chunks.append(
                        {"data": chunk, "label": label, "patient_id": patient_id}
                    )

        return chunks

    def min_max_scale(self, data: np.ndarray) -> np.ndarray:
        """Apply min-max scaling to the input data.

        Args:
            data (np.ndarray): Input data to scale

        Returns:
            np.ndarray: Scaled data between 0 and 1
        """
        return (data - self.min_val) / (self.max_val - self.min_val)

    def legacy_getitem(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - x: HRV signal tensor of shape (n_peaks_per_sample,)
                - label: Class label tensor with values 0, 1, or 2 indicating:
                    0: No complications
                    1: Other complications
                    2: Diabetic peripheral neuropathy
        """
        # Get sample data
        sample = self.raw_data[idx]
        hrv_data = sample["data"]

        # Check if we have enough data points
        if len(hrv_data) < self.n_peaks_per_sample:
            # Pad with repeated data if too short
            hrv_data = np.pad(
                hrv_data, (0, self.n_peaks_per_sample - len(hrv_data)), mode="wrap"
            )

        if self.slice_strategy == "random":
            # Get random chunk of data
            start_idx = random.randint(0, len(hrv_data) - self.n_peaks_per_sample)
            end_idx = start_idx + self.n_peaks_per_sample

        elif self.slice_strategy == "fixed":
            # Get fixed chunk of data
            start_idx = 0
            end_idx = self.n_peaks_per_sample

        x = hrv_data[start_idx:end_idx].astype(np.float32)
        # Apply min-max scaling
        x = self.min_max_scale(x)
        label = sample["class"]

        # Convert to tensors
        x = torch.from_numpy(x).to(torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return (x, label)


def load_dataloaders(config: OmegaConf, fold: int = 0) -> Tuple[DataLoader, DataLoader]:
    """Load the dataloaders for the training and validation sets."""
    assert os.path.exists(config.hrv_data_dir), "HRV data directory does not exist"

    train_dataset = HRVDataset(config, fold, "train")
    val_dataset = HRVDataset(config, fold, "val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=config.train.shuffle,
        num_workers=config.train.num_workers,
        pin_memory=config.train.pin_memory,
        drop_last=config.train.drop_last,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val.batch_size,
        shuffle=config.val.shuffle,
        num_workers=config.val.num_workers,
        pin_memory=config.val.pin_memory,
        drop_last=config.val.drop_last,
    )
    return train_loader, val_loader
