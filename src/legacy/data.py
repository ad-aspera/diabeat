import os
import pickle
import random
from typing import List, Dict, Tuple, Union

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader


class HRVDataset(Dataset):
    """Dataset for handling Heart Rate Variability (HRV) data.

    This dataset loads HRV signals from pickle files, applies preprocessing (cleaning,
    min-max scaling, chunking), and provides access to samples for model training and evaluation.

    The dataset supports different class configurations and chunking strategies:
    - Class configs: 'all' (3 classes), 'diab_v_comp' (2 classes), 'comp_v_dpn' (2 classes)
    - Slice strategies: 'sliding' (overlapping), 'chunked' (non-overlapping)

    Attributes:
        raw_data (List[Dict]): Raw HRV data for each patient
        n_peaks_per_sample (int): Number of HRV peaks to include in each sample
        min_val (float): Minimum HRV value for scaling
        max_val (float): Maximum HRV value for scaling
        slice_strategy (str): Strategy for generating chunks ('sliding' or 'chunked')
        class_config (str): Class configuration ('all', 'diab_v_comp', or 'comp_v_dpn')
        chunks (List[Dict]): Processed data chunks ready for training
    """

    def __init__(self, config: OmegaConf, fold: int = 0, split: str = "train") -> None:
        """Initialize the HRV dataset with the given configuration.

        Args:
            config (OmegaConf): Configuration object containing data parameters
            fold (int, optional): Cross-validation fold to use. Defaults to 0.
            split (str, optional): Data split to use ('train' or 'val'). Defaults to "train".
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
        self.class_based_stride = config.class_based_stride

        self.clean_data()
        self.merge_classes()
        self.chunks = self.generate_chunks()

    def __len__(self) -> int:
        """Return the total number of samples in the dataset.

        Returns:
            int: Number of samples (chunks) in the dataset
        """
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            Tuple[torch.Tensor, torch.Tensor, str]: A tuple containing:
                - data: HRV signal tensor of shape (n_peaks_per_sample,)
                - label: Class label tensor (0, 1, or 2)
                - patient_id: ID of the patient the sample belongs to
        """
        return (
            torch.tensor(self.chunks[idx]["data"], dtype=torch.float32),
            torch.tensor(self.chunks[idx]["label"], dtype=torch.long),
            self.chunks[idx]["patient_id"],
        )

    def clean_data(self) -> None:
        """Clean the data by removing samples with HRV values outside the threshold.

        Removes outlier HRV values that fall outside the defined min and max thresholds.
        Prints a summary of the number of outliers removed.
        """
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
        """Merge classes based on the specified class_config.

        Supports three configurations:
        - 'all': Keep original 3 classes (0: diabetes, 1: complications, 2: DPN)
        - 'diab_v_comp': Binary classification (0: diabetes, 1: complications or DPN)
        - 'comp_v_dpn': Binary classification (0: diabetes or complications, 1: DPN)
        """
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
        """Generate min and max values from the training set for consistent scaling.

        Args:
            hrv_data_dir (str): Directory containing HRV data
            data_config (str): Data configuration name

        Returns:
            Tuple[float, float]: Minimum and maximum HRV values from the training set
        """
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
        """Generate chunks of data based on the specified slice strategy.

        Supports two strategies:
        - 'sliding': Creates overlapping chunks with configurable stride
        - 'chunked': Creates non-overlapping chunks

        Also handles padding for samples shorter than n_peaks_per_sample.

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
                # Get stride based on class label
                stride = self.class_based_stride[label]

                # Sliding window with class-specific stride
                for start_idx in range(
                    0, len(hrv_data) - self.n_peaks_per_sample + 1, stride
                ):
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

        # Count chunks per class
        class_counts = {}
        for chunk in chunks:
            label = chunk["label"]
            class_counts[label] = class_counts.get(label, 0) + 1

        print("\nChunks per class:")
        for label in sorted(class_counts.keys()):
            print(f"Class {label}: {class_counts[label]} chunks")

        return chunks

    def min_max_scale(self, data: np.ndarray) -> np.ndarray:
        """Apply min-max scaling to the input data.

        Scales the input values to the range [0, 1] using the dataset's min and max values.

        Args:
            data (np.ndarray): Input HRV data to scale

        Returns:
            np.ndarray: Scaled data in the range [0, 1]
        """
        return (data - self.min_val) / (self.max_val - self.min_val)

    def legacy_getitem(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Legacy method for retrieving samples using random or fixed slicing.

        This method is preserved for backward compatibility but is not used in the
        current implementation, which uses pre-generated chunks instead.

        Args:
            idx (int): Index of the patient to retrieve data from

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - x: HRV signal tensor of shape (n_peaks_per_sample,)
                - label: Class label tensor (0, 1, or 2)
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
    """Create and return DataLoaders for training and validation sets.

    This function initializes HRVDataset objects for both training and validation,
    and wraps them in DataLoader objects with the specified configuration.

    Args:
        config (OmegaConf): Configuration object containing data and DataLoader parameters
        fold (int, optional): Cross-validation fold to use. Defaults to 0.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing:
            - train_loader: DataLoader for training data
            - val_loader: DataLoader for validation data

    Raises:
        AssertionError: If the specified HRV data directory does not exist
    """
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
