import os
import pickle
import random
import argparse

from process_data import load_data, process_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate k-fold cross validation splits"
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        required=True,
        help="Base directory containing data",
    )
    parser.add_argument(
        "--data_config", type=str, required=True, help="Data configuration name"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Set random seed
    random.seed(args.seed)

    # Load processed data
    input_path = os.path.join(
        args.base_data_dir, args.data_config, f"{args.data_config}_raw.h5"
    )

    features_path = os.path.join(args.base_data_dir, "features.xlsx")
    data_dict, features_df = load_data(input_path, features_path)
    processed_list = process_data(data_dict, features_df)

    # Get class 2 patients for leave-one-out validation
    class_2_patients = [p for p in processed_list if p["class"] == 2]
    k = len(class_2_patients)  # Set k to number of class 2 patients

    print(f"Number of class 2 patients: {k}, using this as number of folds")

    # Create output directory if it doesn't exist
    base_dir = os.path.join(args.base_data_dir, args.data_config)
    os.makedirs(base_dir, exist_ok=True)

    # For each fold
    for fold in range(k):
        # Create fold directory
        fold_dir = os.path.join(base_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        # Initialize train and val lists
        train_data = []
        val_data = []

        # Select validation patient for class 2
        val_class_2 = class_2_patients[fold]

        # Split remaining data
        for class_num in [0, 1, 2]:
            # Get all patients of this class except validation patient
            class_patients = [
                p
                for p in processed_list
                if p["class"] == class_num and p != val_class_2
            ]

            if class_num == 2:
                # For class 2, all remaining patients go to train
                train_data.extend(class_patients)
            else:
                # For class 0 and 1, do 80-20 split
                n_val = max(1, int(len(class_patients) * 0.2))
                val_patients = random.sample(class_patients, n_val)
                train_patients = [p for p in class_patients if p not in val_patients]

                train_data.extend(train_patients)
                val_data.extend(val_patients)

        # Add the class 2 validation patient
        val_data.append(val_class_2)

        # Print validation IDs for class 1 and 2
        print(f"\nFold {fold} validation IDs:")
        print("Class 1:", [p["id"] for p in val_data if p["class"] == 1])
        print("Class 2:", [p["id"] for p in val_data if p["class"] == 2])

        # Save train and validation data
        with open(os.path.join(fold_dir, "train.pkl"), "wb") as f:
            pickle.dump(train_data, f)

        with open(os.path.join(fold_dir, "val.pkl"), "wb") as f:
            pickle.dump(val_data, f)


if __name__ == "__main__":
    main()
