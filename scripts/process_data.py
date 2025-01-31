import os
import pandas as pd
import pickle
from typing import Tuple


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
    data_features_new = []

    idx = 0
    for patient_id, hrv_data in hrv_raw.items():
        # Create a new id entry in data_features_new
        data_features_new = {
            "patient_id": patient_id,
        }

        # Add all HRV features
        for feature in hrv_data.keys():
            data_features_new[idx][feature] = hrv_data[feature]

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
                data_features_new[idx][feature_name] = row[col].values[0]

        idx += 1

    return data_features_new


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
