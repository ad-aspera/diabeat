# Diabeat

A deep learning project for Diabetes Neuropathy Prediction using Heart Rate Variability (HRV) Features. This project uses a transformer-based model to analyze HRV data and predict Diabetic Peripheral Neuropathy (DPN).

## Project Structure

```
diabeat/
├── src/               # Source code
├── data/              # Data directory (to be downloaded)
├── scripts/           # Utility scripts
├── outputs/           # Model outputs and results
├── logs/              # Training logs
├── wandb/             # Weights & Biases logs
├── pyproject.toml     # Project dependencies
└── uv.lock           # UV lock file for dependency management
```

## Data Setup

The project uses HRV data that needs to be downloaded from Zenodo. Follow these steps:

1. Download the data:
```bash
wget https://zenodo.org/records/15053984/files/data.zip
```

2. Unzip the data:
```bash
unzip data.zip -d data/
```

## Installation

This project uses UV for dependency management. Follow these steps to set up the environment:

1. Install UV if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
uv venv
```

## Configuration

The project uses Hydra for configuration management. The main configuration file is located at `src/config.yaml`. Key configuration parameters include:

- Model architecture settings (transformer parameters)
- Data loading and preprocessing options
- Training parameters
- Weights & Biases logging configuration

## Usage

To start a training run, use the following command. Every configuration from the `src/config.yaml` file can be overridden by adding it to the command line. The example below overrides some configurations:
```bash
python src/trainer.py \
model.dim_model=32 \
model.n_heads=4 \
model.dim_feedforward=128 \
model.dropout=0.2 \
data.data_config=nabian_w_index \
data.train.batch_size=256 \
data.val.batch_size=256 \
trainer.epochs=500 \
trainer.optim.use_lr_scheduler=true \
trainer.optim.lr=0.0003 \
trainer.optim.weight_decay=0.01 \
trainer.use_validation=true \
trainer.log_every_n_steps=50 \
trainer.run_validation_every_n_steps=1000 \
wandb.project=<project_name> \
wandb.name=<run_name>
```

Activate the virtual environment before running the command by running `uv venv` and then run the command above.

The scripts are used to process the raw `.h5` data files. You don't need to run them unless you want to preprocess the data differently as all the processed data is already provided in the `data` directory.