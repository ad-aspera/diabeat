import os
import h5py
import pandas as pd
import pickle
import random
from typing import Tuple, List, Dict, Any
import argparse


def get_datasets(name: str, obj: Any) -> dict:
    """
    Recursively get all datasets from an HDF5 file
    Args:
        name: Name of the dataset/group
        obj: HDF5 dataset/group object
    Returns:
        Dictionary containing dataset name and values
    """
    datasets = {}
    if isinstance(obj, h5py.Dataset):
        # Extract just the ID number from the full path
        key = name.split("/")[0].split("_")[0]
        datasets[key] = obj[:]
    elif isinstance(obj, h5py.Group):
        for key, val in obj.items():
            # Recursively get datasets from nested groups
            datasets.update(get_datasets(f"{name}/{key}" if name else key, val))
    return datasets


def load_data(h5_path: str, features_path: str) -> Tuple[dict, pd.DataFrame]:
    """
    Load HRV data from H5 file and features from Excel file

    Args:
        h5_path (str): Path to the HRV H5 file
        features_path (str): Path to the features Excel file

    Returns:
        tuple: (hrv_data, features_df) containing the loaded data
    """
    # Load HRV data from H5 file
    data_dict = {}
    with h5py.File(h5_path, "r") as f:
        data_dict = get_datasets("", f)

    # Load features from Excel file
    features_df = pd.read_excel(features_path)

    return data_dict, features_df


def process_data(data_dict: dict, features_df: pd.DataFrame) -> List[Dict]:
    """
    Process and combine HRV and features data

    Args:
        data_dict (dict): Raw HRV data dictionary
        features_df (pd.DataFrame): Features dataframe

    Returns:
        List[Dict]: List of processed data combining HRV and class labels
    """
    processed_list = []

    # Iterate through data_dict
    for id_, data in data_dict.items():
        # Find matching row in features_df
        feature_row = features_df[features_df["Unnamed: 0"] == int(id_)]

        if not feature_row.empty:
            # Get class label based on conditions
            if feature_row["Diabetic Complications"].iloc[0] == 0:
                class_label = 0
            elif feature_row["Diabetic peripheral neuropathy"].iloc[0] == 1:
                class_label = 2
            else:
                class_label = 1

            # Store data and class in dictionary with id
            processed_list.append({"id": id_, "data": data, "class": class_label})

    return processed_list


def create_train_val_split(
    processed_list: List[Dict], train_ratio: float = 0.8, random_seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Create stratified train and validation splits

    Args:
        processed_list (List[Dict]): Processed data list
        train_ratio (float): Ratio of training data (default: 0.8)
        random_seed (int): Random seed for reproducibility (default: 42)

    Returns:
        Tuple[List[Dict], List[Dict]]: Tuple of (train_list, val_list)
    """
    random.seed(random_seed)

    # Separate data by class
    class_data = {0: [], 1: [], 2: []}
    for item in processed_list:
        class_data[item["class"]].append(item)

    # Create train and validation splits for each class
    train_list = []
    val_list = []

    for class_label in [0, 1, 2]:
        class_items = class_data[class_label]
        n_samples = len(class_items)

        # Ensure at least 1 sample in each split
        if n_samples == 1:
            # If only 1 sample, duplicate it for both splits
            train_list.append(class_items[0])
            val_list.append(class_items[0])
        else:
            # Calculate split sizes
            n_train = max(int(train_ratio * n_samples), 1)  # At least 1 for train
            n_val = max(n_samples - n_train, 1)  # At least 1 for val

            # Randomly shuffle the items
            random.shuffle(class_items)

            # Split the data
            train_list.extend(class_items[:n_train])
            val_list.extend(class_items[n_train:])

    return train_list, val_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process HRV data and create train/val splits"
    )
    parser.add_argument("--h5_path", type=str, help="Path to the HRV H5 file")
    parser.add_argument(
        "--features_path", type=str, help="Path to the features Excel file"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Directory to save the processed data"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of training data (default: 0.8)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Use provided paths or default to relative paths
    if args.h5_path and args.features_path and args.output_dir:
        h5_path = args.h5_path
        features_path = args.features_path
        data_dir = args.output_dir
    else:
        # Get parent directory path
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        h5_path = os.path.join(parent_dir, "data", "ml_data.h5")
        features_path = os.path.join(parent_dir, "data", "features.xlsx")
        data_dir = os.path.join(parent_dir, "data")

    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Load data
    data_dict, features_df = load_data(h5_path, features_path)

    # Process data
    processed_list = process_data(data_dict, features_df)

    # Create train and validation splits
    train_list, val_list = create_train_val_split(
        processed_list, train_ratio=args.train_ratio, random_seed=args.random_seed
    )

    # Save train and validation splits
    train_path = os.path.join(data_dir, "train.pkl")
    val_path = os.path.join(data_dir, "val.pkl")

    with open(train_path, "wb") as f:
        pickle.dump(train_list, f)
    with open(val_path, "wb") as f:
        pickle.dump(val_list, f)

    print(f"Saved train.pkl and val.pkl to {data_dir}")
    print("\nTraining set class distribution:")
    train_counts = {0: 0, 1: 0, 2: 0}
    for item in train_list:
        train_counts[item["class"]] += 1
    print(f"Class 0 (No complications): {train_counts[0]}")
    print(f"Class 1 (Other complications): {train_counts[1]}")
    print(f"Class 2 (Diabetic peripheral neuropathy): {train_counts[2]}")

    print("\nValidation set class distribution:")
    val_counts = {0: 0, 1: 0, 2: 0}
    for item in val_list:
        val_counts[item["class"]] += 1
    print(f"Class 0 (No complications): {val_counts[0]}")
    print(f"Class 1 (Other complications): {val_counts[1]}")
    print(f"Class 2 (Diabetic peripheral neuropathy): {val_counts[2]}")
