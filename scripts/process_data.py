import os
import pandas as pd
import pickle
import random
from typing import Tuple, List, Dict, Any


def load_data(hrv_path: str, features_path: str) -> Tuple[dict, pd.DataFrame]:
    """
    Load HRV and features data from files

    Args:
        hrv_path (str): Path to the HRV pickle file
        features_path (str): Path to the features Excel file

    Returns:
        tuple: (hrv_raw, features_df) containing the loaded data
    """
    with open(hrv_path, "rb") as f:
        hrv_raw = pickle.load(f)

    features_df = pd.read_excel(features_path)

    return hrv_raw, features_df


def process_data(hrv_raw: dict, features_df: pd.DataFrame) -> dict:
    """
    Process and combine HRV and features data

    Args:
        hrv_raw (dict): Raw HRV data dictionary
        features_df (pd.DataFrame): Features dataframe

    Returns:
        dict: Processed data combining HRV and features
    """
    data_features = []

    for patient_id, hrv_data in hrv_raw.items():
        # Create a new id entry in data_feature
        data_feature = {
            "patient_id": patient_id,
        }

        # Add all HRV features
        for feature in hrv_data.keys():
            data_feature[feature] = hrv_data[feature]

        # Add features from features_df
        row = features_df.loc[features_df["Unnamed: 0"] == int(patient_id)]

        # Add all columns except Unnamed: 0 as features
        for col in features_df.columns:
            if col != "Unnamed: 0":
                # Clean up column name by removing parentheses and content inside them
                clean_col = col.strip()
                if "(" in clean_col:
                    clean_col = clean_col[: clean_col.index("(")].strip()

                # Create standardized feature name
                feature_name = clean_col.lower().strip()
                feature_name = feature_name.replace(" ", "_")
                feature_name = feature_name.replace("-", "_")
                feature_name = feature_name.replace("%", "")
                feature_name = feature_name.replace("/", "_")

                # Add feature to dictionary
                data_feature[feature_name] = row[col].values[0]

        data_features.append(data_feature)

    return data_features


def create_train_val_split(
    samples: List[Dict[str, Any]], train_val_ratio: float = 0.8, random_seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Create train and validation splits based on patient IDs while ensuring patients
    with neuropathy are distributed across both sets.

    Args:
        samples (List[Dict[str, Any]]): List of dictionaries containing patient data
        train_val_ratio (float): Ratio of training data (default: 0.8)
        random_seed (int): Random seed for reproducibility (default: 42)

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: Tuple of (train_samples, val_samples)
    """
    # Get patient IDs with and without neuropathy
    patient_ids_with_neuropathy = set(
        sample["patient_id"]
        for sample in samples
        if int(sample["diabetic_peripheral_neuropathy"]) == 1
    )

    patient_ids_without_neuropathy = set(
        sample["patient_id"]
        for sample in samples
        if int(sample["diabetic_peripheral_neuropathy"]) == 0
    )

    # Convert to sorted lists for reproducibility
    neuropathy_ids = sorted(list(patient_ids_with_neuropathy))
    non_neuropathy_ids = sorted(list(patient_ids_without_neuropathy))

    # Set random seed and shuffle non-neuropathy IDs
    random.seed(random_seed)
    random.shuffle(non_neuropathy_ids)

    # Split non-neuropathy IDs
    split_idx = int(train_val_ratio * len(non_neuropathy_ids))
    train_non_neuropathy_ids = non_neuropathy_ids[:split_idx]
    val_non_neuropathy_ids = non_neuropathy_ids[split_idx:]

    # Split neuropathy IDs (hardcoded split as in notebook)
    train_neuropathy_ids = ["20010826", "20101822", "20123017"]
    val_neuropathy_ids = ["19101619"]

    # Combine IDs for final splits
    train_ids = train_non_neuropathy_ids + train_neuropathy_ids
    val_ids = val_non_neuropathy_ids + val_neuropathy_ids

    # Create train and validation datasets
    train_samples = [sample for sample in samples if sample["patient_id"] in train_ids]
    val_samples = [sample for sample in samples if sample["patient_id"] in val_ids]

    return train_samples, val_samples


if __name__ == "__main__":
    # This assumes that the script is run from the root directory of the project
    # and that the data is stored in the data/ directory

    # Get parent directory path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Construct data paths relative to parent directory
    hrv_path = os.path.join(parent_dir, "data", "data.pkl")
    features_path = os.path.join(parent_dir, "data", "features.xlsx")
    processed_data_path = os.path.join(parent_dir, "data", "processed.pkl")

    # Load data
    hrv_raw, features_df = load_data(hrv_path, features_path)

    # Process data
    processed_data = process_data(hrv_raw, features_df)

    # Save processed data to pickle file
    with open(processed_data_path, "wb") as f:
        pickle.dump(processed_data, f)

    # Create train and validation splits
    train_samples, val_samples = create_train_val_split(processed_data)

    train_path = os.path.join(parent_dir, "data", "train.pkl")
    val_path = os.path.join(parent_dir, "data", "val.pkl")

    # Save train and validation splits to pickle files
    with open(train_path, "wb") as f:
        pickle.dump(train_samples, f)
    with open(val_path, "wb") as f:
        pickle.dump(val_samples, f)
