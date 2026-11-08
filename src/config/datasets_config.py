"""
Dataset Configuration Module.

This module centralizes the paths for all datasets used in the experiment.
It uses 'pathlib' to ensure cross-platform compatibility and dynamic 
path resolution relative to the project's root directory.

Attributes:
    BASE_DIR (Path): The absolute path to the project's root directory.
    DATA_DIR (Path): The directory where dataset files are stored.
    DATASETS (dict): A mapping of dataset names to their respective file paths.
"""

from pathlib import Path
from typing import Dict

# Path resolution: Assumes this file is in src/config/
# .parent.parent points to the project root (PyGAwLLfs2/)
BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
DATA_DIR: Path = BASE_DIR / "data"

# DATASETS: Dict[str, Path] = {
#     "boson": DATA_DIR / "boson.dat",
#     "zoo": DATA_DIR / "zoo.dat",
#     "artificial": DATA_DIR / "artificial.dat",
#     "ionosphere": DATA_DIR / "ionosphere.dat",
#     "libras": DATA_DIR / "libras.dat",
#     "sonar": DATA_DIR / "sonar.dat",
#     "covidx": DATA_DIR / "covidx.dat"
# }

DATASETS: Dict[str, Path] =  {
    "Corn-22": DATA_DIR / "Corn/Corn_reduzido_22.dat",
    'Corn-23': DATA_DIR / "Corn/Corn_reduzido_23.dat",
    'Corn-25': DATA_DIR / "Corn/Corn_reduzido_25.dat",
    'Corn-26': DATA_DIR / 'Corn/Corn_reduzido_26.dat',
    'Corn-27': DATA_DIR / 'Corn/Corn_reduzido_27.dat',
    'Corn-29': DATA_DIR / 'Corn/Corn_reduzido_29.dat',
    'Soy-22': DATA_DIR / 'Soy/Soy_reduzido_22.dat',
    'Soy-23': DATA_DIR / 'Soy/Soy_reduzido_23.dat',
    'Soy-25': DATA_DIR / 'Soy/Soy_reduzido_25.dat',
    'Soy-26': DATA_DIR / 'Soy/Soy_reduzido_26.dat',
    'Soy-27': DATA_DIR / 'Soy/Soy_reduzido_27.dat',
    'Soy-28': DATA_DIR / 'Soy/Soy_reduzido_28.dat',
    'Soy-29': DATA_DIR / 'Soy/Soy_reduzido_29.dat',
    'Cotton-22': DATA_DIR / 'Cotton/Cotton_reduzido_22.dat',
    'Cotton-23': DATA_DIR / 'Cotton/Cotton_reduzido_23.dat',
    'Cotton-25': DATA_DIR / 'Cotton/Cotton_reduzido_25.dat',
    'Cotton-26': DATA_DIR / 'Cotton/Cotton_reduzido_26.dat',
    'Cotton-27': DATA_DIR / 'Cotton/Cotton_reduzido_27.dat',
    'Cotton-29': DATA_DIR / 'Cotton/Cotton_reduzido_29.dat'
}

def get_dataset_path(dataset_name: str) -> Path:
    """
    Retrieves the absolute path for a specific dataset and verifies its existence.

    Args:
        dataset_name (str): The name of the dataset to find.

    Returns:
        Path: The validated Path object for the dataset.

    Raises:
        FileNotFoundError: If the dataset name is not in the configuration
            or if the physical file is missing on the disk.
    """
    if dataset_name not in DATASETS:
        raise FileNotFoundError(f"Dataset '{dataset_name}' not found in configuration.")

    path = DATASETS[dataset_name]

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at: {path}\n"
            f"Check if the file is placed in the '{DATA_DIR}' directory."
        )

    return path