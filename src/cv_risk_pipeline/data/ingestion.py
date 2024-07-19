"""
Data ingestion module for cardiovascular risk prediction pipeline.

This module handles loading and initial processing of clinical data from
various sources including EHR systems and clinical trial databases.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging
from datetime import datetime
import yaml

logger = logging.getLogger(__name__)


class DataIngestion:
    """
    Handles data ingestion from various clinical data sources.
    
    Supports multiple file formats and data structures commonly found
    in EHR systems and clinical trial databases.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data ingestion with configuration.
        
        Args:
            config: Configuration dictionary containing data paths and settings
        """
        self.config = config
        self.data_paths = config.get('data', {})
        self.raw_data_path = Path(self.data_paths.get('raw_data_path', 'data/raw'))
        self.processed_data_path = Path(self.data_paths.get('processed_data_path', 'data/processed'))
        
        # Ensure directories exist
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Data ingestion initialized with raw path: {self.raw_data_path}")
    
    def load_ehr_data(self, file_path: Union[str, Path], 
                     file_format: str = 'auto') -> pd.DataFrame:
        """
        Load EHR data from various file formats.
        
        Args:
            file_path: Path to the data file
            file_format: File format ('csv', 'excel', 'parquet', 'json', 'auto')
            
        Returns:
            DataFrame containing the loaded data
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file does not exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Auto-detect format if not specified
        if file_format == 'auto':
            file_format = file_path.suffix.lower().lstrip('.')
        
        logger.info(f"Loading EHR data from {file_path} (format: {file_format})")
        
        try:
            if file_format in ['csv', 'tsv']:
                delimiter = '\t' if file_format == 'tsv' else ','
                df = pd.read_csv(file_path, delimiter=delimiter, low_memory=False)
            elif file_format in ['xlsx', 'xls']:
                df = pd.read_excel(file_path, engine='openpyxl')
            elif file_format == 'parquet':
                df = pd.read_parquet(file_path)
            elif file_format == 'json':
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            logger.info(f"Successfully loaded {len(df)} records from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
    
    def load_clinical_trial_data(self, file_path: Union[str, Path],
                               study_id: Optional[str] = None) -> pd.DataFrame:
        """
        Load clinical trial data with study-specific processing.
        
        Args:
            file_path: Path to the clinical trial data file
            study_id: Optional study identifier for metadata
            
        Returns:
            DataFrame containing the clinical trial data
        """
        df = self.load_ehr_data(file_path)
        
        # Add study metadata if provided
        if study_id:
            df['study_id'] = study_id
            df['data_source'] = 'clinical_trial'
        else:
            df['data_source'] = 'clinical_trial'
        
        logger.info(f"Loaded clinical trial data with {len(df)} records")
        return df
    
    def load_longitudinal_data(self, base_path: Union[str, Path],
                             patient_id_col: str = 'patient_id',
                             date_col: str = 'date') -> pd.DataFrame:
        """
        Load longitudinal clinical data from multiple files.
        
        Args:
            base_path: Base directory containing longitudinal data files
            patient_id_col: Name of the patient ID column
            date_col: Name of the date column
            
        Returns:
            Combined DataFrame with longitudinal data
        """
        base_path = Path(base_path)
        
        if not base_path.exists():
            raise FileNotFoundError(f"Base path not found: {base_path}")
        
        # Find all data files
        data_files = []
        for pattern in ['*.csv', '*.xlsx', '*.parquet']:
            data_files.extend(base_path.glob(pattern))
        
        if not data_files:
            raise FileNotFoundError(f"No data files found in {base_path}")
        
        logger.info(f"Found {len(data_files)} data files in {base_path}")
        
        # Load and combine all files
        combined_data = []
        for file_path in data_files:
            try:
                df = self.load_ehr_data(file_path)
                df['source_file'] = file_path.name
                combined_data.append(df)
                logger.info(f"Loaded {len(df)} records from {file_path.name}")
            except Exception as e:
                logger.warning(f"Failed to load {file_path.name}: {str(e)}")
                continue
        
        if not combined_data:
            raise ValueError("No data files could be loaded successfully")
        
        # Combine all dataframes
        final_df = pd.concat(combined_data, ignore_index=True, sort=False)
        
        # Ensure required columns exist
        if patient_id_col not in final_df.columns:
            raise ValueError(f"Patient ID column '{patient_id_col}' not found in data")
        
        if date_col not in final_df.columns:
            raise ValueError(f"Date column '{date_col}' not found in data")
        
        # Convert date column to datetime
        final_df[date_col] = pd.to_datetime(final_df[date_col], errors='coerce')
        
        # Sort by patient ID and date
        final_df = final_df.sort_values([patient_id_col, date_col])
        
        logger.info(f"Combined longitudinal data: {len(final_df)} total records")
        return final_df
    
    def validate_data_structure(self, df: pd.DataFrame,
                              required_columns: List[str]) -> Dict[str, Any]:
        """
        Validate the structure of loaded data.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'is_valid': True,
            'missing_columns': [],
            'data_types': {},
            'missing_values': {},
            'duplicate_records': 0,
            'total_records': len(df)
        }
        
        # Check for required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['missing_columns'] = missing_cols
            validation_results['is_valid'] = False
            logger.warning(f"Missing required columns: {missing_cols}")
        
        # Check data types
        for col in df.columns:
            validation_results['data_types'][col] = str(df[col].dtype)
        
        # Check for missing values
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            validation_results['missing_values'][col] = {
                'count': missing_count,
                'percentage': missing_pct
            }
        
        # Check for duplicate records
        validation_results['duplicate_records'] = df.duplicated().sum()
        
        logger.info(f"Data validation completed. Valid: {validation_results['is_valid']}")
        return validation_results
    
    def save_processed_data(self, df: pd.DataFrame, 
                          filename: str,
                          format: str = 'parquet') -> Path:
        """
        Save processed data to the processed data directory.
        
        Args:
            df: DataFrame to save
            filename: Name of the output file (without extension)
            format: Output format ('parquet', 'csv', 'excel')
            
        Returns:
            Path to the saved file
        """
        output_path = self.processed_data_path / f"{filename}.{format}"
        
        try:
            if format == 'parquet':
                df.to_parquet(output_path, index=False)
            elif format == 'csv':
                df.to_csv(output_path, index=False)
            elif format == 'excel':
                df.to_excel(output_path, index=False, engine='openpyxl')
            else:
                raise ValueError(f"Unsupported output format: {format}")
            
            logger.info(f"Saved processed data to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving data to {output_path}: {str(e)}")
            raise
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the loaded data.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary containing data summary statistics
        """
        summary = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'date_range': {},
            'numeric_columns': [],
            'categorical_columns': [],
            'missing_data_summary': {}
        }
        
        # Date range analysis
        date_columns = df.select_dtypes(include=['datetime64']).columns
        for col in date_columns:
            if not df[col].isnull().all():
                summary['date_range'][col] = {
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'span_days': (df[col].max() - df[col].min()).days
                }
        
        # Column type analysis
        summary['numeric_columns'] = df.select_dtypes(include=[np.number]).columns.tolist()
        summary['categorical_columns'] = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Missing data summary
        missing_summary = df.isnull().sum()
        summary['missing_data_summary'] = {
            'columns_with_missing': (missing_summary > 0).sum(),
            'total_missing_values': missing_summary.sum(),
            'missing_percentage': (missing_summary.sum() / (len(df) * len(df.columns))) * 100
        }
        
        logger.info(f"Data summary: {summary['total_records']} records, "
                   f"{summary['total_columns']} columns, "
                   f"{summary['missing_data_summary']['missing_percentage']:.1f}% missing data")
        
        return summary



