import os
from omegaconf import OmegaConf
from typing import Union
from pathlib import Path


# obtain the folder path of the current file
folder_path = Path(__file__).parent


def load_config(
    config_id: str,
    config_path: Union[str, Path] = os.path.join(folder_path, "config.yaml"),
) -> OmegaConf:
    """
    Load configuration from a YAML file using OmegaConf.

    Args:
        config_id (str): The ID of the configuration to load. Select from ["model", "data", "trainer"]
        config_path (Union[str, Path]): Path to the YAML configuration file.
            defaults to "config.yaml" in the src directory.

    Returns:
        OmegaConf: Configuration object

    Raises:
        FileNotFoundError: If the config file doesn't exist
        ValueError: If the config file is invalid
    """
    config_path = Path(config_path)

    assert config_id in ["model", "data", "trainer"], "Invalid config ID"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    try:
        config = OmegaConf.load(config_path)
        return config[config_id]
    except Exception as e:
        raise ValueError(f"Failed to load config file: {str(e)}")
