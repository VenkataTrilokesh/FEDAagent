"""AI Data Scientist package.

End-to-end toolkit for data profiling, cleaning, EDA, baseline model training,
explainability (SHAP/LIME), notebook generation, and report exports.
"""

from .data_loader import DatasetLoader, DatasetProfile
from .preprocessing import DataCleaner
from .eda import EDAEngine
from .imbalance_handler import ImbalanceHandler
from .notebook_generator import NotebookGenerator
from .modeling import BaselineModelTrainer
from .explainability import ExplainabilityEngine
from .reporting import ReportBuilder
from .pipeline import FullPipeline

__all__ = [
    "DatasetLoader",
    "DatasetProfile",
    "DataCleaner",
    "EDAEngine",
    "ImbalanceHandler",
    "NotebookGenerator",
    "BaselineModelTrainer",
    "ExplainabilityEngine",
    "ReportBuilder",
    "FullPipeline",
]
