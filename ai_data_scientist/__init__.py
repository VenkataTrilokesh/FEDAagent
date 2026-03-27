"""AI Data Scientist package.

A modular, production-friendly toolkit that profiles tabular datasets,
cleans them, performs advanced EDA, handles class imbalance, and generates
fully structured Jupyter notebooks.
"""

from .data_loader import DatasetLoader, DatasetProfile
from .preprocessing import DataCleaner
from .eda import EDAEngine
from .imbalance_handler import ImbalanceHandler
from .notebook_generator import NotebookGenerator

__all__ = [
    "DatasetLoader",
    "DatasetProfile",
    "DataCleaner",
    "EDAEngine",
    "ImbalanceHandler",
    "NotebookGenerator",
]
