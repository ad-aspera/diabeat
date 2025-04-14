import os
import pickle
import random
from typing import List, Dict, Tuple, Union, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader


class HRVPeaksDataset(Dataset):

    def __init__(
        self, config: OmegaConf, split: str = "train", shuffle: bool = False
    ) -> None:
        """Initialize the HRV dataset with the given configuration.

        Args:
            config (OmegaConf): Configuration object containing data parameters
            split (str, optional): Data split to use ('train' or 'val'). Defaults to "train".
            shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
        """
        self.split = split
        self.shuffle = shuffle
        # Load raw HRV data
        chunks_path = os.path.join(
            config.hrv_data_dir,
            config.data_config,
            f"{split}.pkl",
        )
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

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
            torch.tensor(self.chunks[idx]["data"], dtype=torch.long),
            torch.tensor(self.chunks[idx]["label"], dtype=torch.long),
            self.chunks[idx]["patient_id"],
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
            # Merge complications (1) and DPN (2) into class 1
            for patient in self.raw_data:
                if patient["class"] == 2:
                    patient["class"] = 1
        elif self.class_config == "comp_v_dpn":
            # Merge diabetes (0) and complications (1) into 0, and DPN (2) -> 1
            for patient in self.raw_data:
                if patient["class"] == 2:
                    patient["class"] = 1
                elif patient["class"] == 1:
                    patient["class"] = 0

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
        chunks = []
        rejected_chunks = 0
        for patient in self.raw_data:
            hrv_data = patient["data"]
            label = patient["class"]
            patient_id = patient["id"]

            # Get stride based on class label
            stride = self.class_based_stride[label]

            # Sliding window with class-specific stride
            for start_idx in range(
                0, len(hrv_data) - self.n_peaks_per_sample + 1, stride
            ):
                chunk = hrv_data[start_idx : start_idx + self.n_peaks_per_sample]
                # assess whether the chunk is valid
                validated_chunk = self.validate_chunk(chunk)
                if validated_chunk is not None:
                    chunks.append(
                        {
                            "data": validated_chunk,
                            "label": label,
                            "patient_id": patient_id,
                        }
                    )
                else:
                    rejected_chunks += 1
        print(f"Rejected {rejected_chunks} chunks")
        return chunks

    def validate_chunk(self, chunk: np.ndarray) -> bool:
        """Validate a chunk of HRV data.

        Args:
            chunk (np.ndarray): The chunk of HRV data to validate

        """
        # units here are 10 milliseconds (not 1 millisecond)
        # make the first value 0
        chunk = chunk - chunk[0]
        # convert to 10 ms
        chunk = chunk // 10

        # ensure that heart rate (beats per minute) is within the range
        hr = self.n_peaks_per_sample / (chunk[-1] / (6000))
        if hr >= self.max_heart_rate or hr <= self.min_heart_rate:
            return None
        else:
            return chunk

    def split_data_after_generating_chunks(
        self,
    ) -> List[Dict[str, Union[np.ndarray, int, str]]]:
        """Split the data into train and validation sets."""
        if self.shuffle:
            random.shuffle(self.chunks)

        # Split chunks by class
        class_0_chunks = [chunk for chunk in self.chunks if chunk["label"] == 0]
        class_1_chunks = [chunk for chunk in self.chunks if chunk["label"] == 1]

        # Take 80% from each class
        train_class_0 = class_0_chunks[: int(len(class_0_chunks) * 0.8)]
        train_class_1 = class_1_chunks[: int(len(class_1_chunks) * 0.8)]

        # Take remaining 20% from each class
        val_class_0 = class_0_chunks[int(len(class_0_chunks) * 0.8) :]
        val_class_1 = class_1_chunks[int(len(class_1_chunks) * 0.8) :]
        return (train_class_0 + train_class_1), (val_class_0 + val_class_1)


def get_fixed_batches(loader: DataLoader, n_batches: Optional[int] = None) -> List:
    """Get a fixed set of batches from a DataLoader for consistent training.

    This function extracts a specified number of batches from a DataLoader,
    which can be used to ensure consistent data across epochs.

    Args:
        loader (DataLoader): The data loader to extract batches from
        n_batches (Optional[int], optional): Number of batches to extract, or None for all. Defaults to None.
    Returns:
        List: List of batches, where each batch is a tuple of (inputs, targets, [optional metadata])
    """
    batches = []
    for i, batch in enumerate(loader):
        if n_batches and i >= n_batches:
            break
        batches.append(batch)
    return batches


def load_dataloaders(
    config: OmegaConf,
    n_batches_train: int = None,
    n_batches_val: int = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create and return DataLoaders for training and validation sets.

    This function initializes HRVDataset objects for both training and validation,
    and wraps them in DataLoader objects with the specified configuration.

    Args:
        config (OmegaConf): Configuration object containing data and DataLoader parameters

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing:
            - train_loader: DataLoader for training data
            - val_loader: DataLoader for validation data

    Raises:
        AssertionError: If the specified HRV data directory does not exist
    """
    assert os.path.exists(config.hrv_data_dir), "HRV data directory does not exist"

    # load train and val datasets
    train_dataset = HRVPeaksDataset(config, "train", shuffle=config.train.shuffle)
    val_dataset = HRVPeaksDataset(config, "val", shuffle=config.val.shuffle)

    # create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=config.train.shuffle,
        num_workers=config.train.num_workers,
        # pin_memory=config.train.pin_memory,
        drop_last=config.train.drop_last,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val.batch_size,
        shuffle=config.val.shuffle,
        num_workers=config.val.num_workers,
        # pin_memory=config.val.pin_memory,
        drop_last=config.val.drop_last,
    )

    if n_batches_train is not None:
        train_loader = get_fixed_batches(train_loader, n_batches_train)

    if n_batches_val is not None:
        val_loader = get_fixed_batches(val_loader, n_batches_val)

    return train_loader, val_loader
