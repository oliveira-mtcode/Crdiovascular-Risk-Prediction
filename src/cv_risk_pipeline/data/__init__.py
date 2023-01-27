"""
Data handling modules for cardiovascular risk prediction pipeline.

This module provides data ingestion, cleaning, and preprocessing capabilities
for clinical data including EHR and clinical trial datasets.
"""

from .ingestion import DataIngestion
from .cleaning import DataCleaning
from .preprocessing import DataPreprocessor

__all__ = ["DataIngestion", "DataCleaning", "DataPreprocessor"]
