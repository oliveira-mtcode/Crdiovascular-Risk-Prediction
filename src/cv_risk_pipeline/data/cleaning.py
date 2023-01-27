"""
Data cleaning module for cardiovascular risk prediction pipeline.

This module provides comprehensive data cleaning capabilities for clinical data,
including outlier detection, missing value handling, and data quality assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings

logger = logging.getLogger(__name__)


class DataCleaning:
    """
    Handles data cleaning operations for clinical data.
    
    Provides methods for outlier detection, missing value handling,
    and data quality assessment specific to clinical datasets.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data cleaning with configuration.
        
        Args:
            config: Configuration dictionary containing cleaning parameters
        """
        self.config = config
        self.cleaning_config = config.get('data_cleaning', {})
        self.outlier_config = self.cleaning_config.get('outlier_detection', {})
        self.missing_config = self.cleaning_config.get('missing_values', {})
        self.temporal_config = self.cleaning_config.get('temporal', {})
        
        logger.info("Data cleaning initialized")
    
    def detect_outliers(self, df: pd.DataFrame, 
                       columns: Optional[List[str]] = None,
                       method: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect outliers in clinical data using multiple methods.
        
        Args:
            df: DataFrame containing the data
            columns: List of columns to analyze (if None, uses numeric columns)
            method: Outlier detection method ('iqr', 'zscore', 'isolation_forest')
            
        Returns:
            Dictionary containing outlier detection results
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if method is None:
            method = self.outlier_config.get('method', 'iqr')
        
        logger.info(f"Detecting outliers using {method} method for {len(columns)} columns")
        
        outlier_results = {
            'method': method,
            'columns_analyzed': columns,
            'outlier_indices': [],
            'outlier_summary': {},
            'cleaned_data': df.copy()
        }
        
        all_outlier_indices = set()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found in data")
                continue
            
            # Remove missing values for outlier detection
            non_null_data = df[col].dropna()
            if len(non_null_data) == 0:
                logger.warning(f"No non-null data in column {col}")
                continue
            
            col_outliers = self._detect_column_outliers(non_null_data, col, method)
            
            if col_outliers:
                outlier_results['outlier_summary'][col] = {
                    'count': len(col_outliers),
                    'percentage': (len(col_outliers) / len(non_null_data)) * 100,
                    'indices': col_outliers
                }
                all_outlier_indices.update(col_outliers)
        
        outlier_results['outlier_indices'] = list(all_outlier_indices)
        outlier_results['total_outliers'] = len(all_outlier_indices)
        
        logger.info(f"Detected {len(all_outlier_indices)} outlier records")
        return outlier_results
    
    def _detect_column_outliers(self, data: pd.Series, column: str, method: str) -> List[int]:
        """
        Detect outliers in a single column using specified method.
        
        Args:
            data: Series containing the data
            column: Name of the column
            method: Detection method
            
        Returns:
            List of outlier indices
        """
        outliers = []
        
        if method == 'iqr':
            outliers = self._iqr_outliers(data)
        elif method == 'zscore':
            outliers = self._zscore_outliers(data)
        elif method == 'isolation_forest':
            outliers = self._isolation_forest_outliers(data)
        else:
            raise ValueError(f"Unsupported outlier detection method: {method}")
        
        return outliers
    
    def _iqr_outliers(self, data: pd.Series) -> List[int]:
        """Detect outliers using Interquartile Range method."""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        multiplier = self.outlier_config.get('iqr_multiplier', 1.5)
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)].index.tolist()
        return outliers
    
    def _zscore_outliers(self, data: pd.Series) -> List[int]:
        """Detect outliers using Z-score method."""
        threshold = self.outlier_config.get('zscore_threshold', 3.0)
        z_scores = np.abs(stats.zscore(data))
        outliers = data[z_scores > threshold].index.tolist()
        return outliers
    
    def _isolation_forest_outliers(self, data: pd.Series) -> List[int]:
        """Detect outliers using Isolation Forest method."""
        # Reshape data for sklearn
        data_reshaped = data.values.reshape(-1, 1)
        
        # Fit isolation forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels = iso_forest.fit_predict(data_reshaped)
        
        # Get outlier indices
        outliers = data[outlier_labels == -1].index.tolist()
        return outliers
    
    def handle_missing_values(self, df: pd.DataFrame,
                            strategy: Optional[str] = None,
                            max_missing_pct: Optional[float] = None) -> pd.DataFrame:
        """
        Handle missing values in clinical data.
        
        Args:
            df: DataFrame containing the data
            strategy: Missing value handling strategy
            max_missing_pct: Maximum percentage of missing values allowed per column
            
        Returns:
            DataFrame with missing values handled
        """
        if strategy is None:
            strategy = self.missing_config.get('strategy', 'interpolate')
        
        if max_missing_pct is None:
            max_missing_pct = self.missing_config.get('max_missing_percentage', 0.3)
        
        logger.info(f"Handling missing values using {strategy} strategy")
        
        df_cleaned = df.copy()
        
        # Remove columns with too many missing values
        missing_pct = df_cleaned.isnull().sum() / len(df_cleaned)
        high_missing_cols = missing_pct[missing_pct > max_missing_pct].index.tolist()
        
        if high_missing_cols:
            logger.warning(f"Removing columns with >{max_missing_pct*100}% missing values: {high_missing_cols}")
            df_cleaned = df_cleaned.drop(columns=high_missing_cols)
        
        # Handle remaining missing values
        if strategy == 'drop':
            df_cleaned = df_cleaned.dropna()
        elif strategy == 'interpolate':
            df_cleaned = self._interpolate_missing_values(df_cleaned)
        elif strategy == 'forward_fill':
            df_cleaned = df_cleaned.fillna(method='ffill')
        elif strategy == 'backward_fill':
            df_cleaned = df_cleaned.fillna(method='bfill')
        else:
            raise ValueError(f"Unsupported missing value strategy: {strategy}")
        
        logger.info(f"Missing value handling completed. "
                   f"Remaining missing values: {df_cleaned.isnull().sum().sum()}")
        
        return df_cleaned
    
    def _interpolate_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate missing values using appropriate methods for different data types.
        
        Args:
            df: DataFrame with missing values
            
        Returns:
            DataFrame with interpolated values
        """
        df_interpolated = df.copy()
        
        for col in df_interpolated.columns:
            if df_interpolated[col].dtype in ['int64', 'float64']:
                # For numeric columns, use linear interpolation
                df_interpolated[col] = df_interpolated[col].interpolate(method='linear')
            elif df_interpolated[col].dtype == 'datetime64[ns]':
                # For datetime columns, use linear interpolation
                df_interpolated[col] = df_interpolated[col].interpolate(method='linear')
            else:
                # For categorical columns, use forward fill
                df_interpolated[col] = df_interpolated[col].fillna(method='ffill')
        
        return df_interpolated
    
    def clean_temporal_data(self, df: pd.DataFrame,
                          patient_id_col: str = 'patient_id',
                          date_col: str = 'date') -> pd.DataFrame:
        """
        Clean temporal clinical data by handling gaps and ensuring minimum observations.
        
        Args:
            df: DataFrame with temporal data
            patient_id_col: Name of the patient ID column
            date_col: Name of the date column
            
        Returns:
            Cleaned DataFrame with temporal data
        """
        logger.info("Cleaning temporal data")
        
        df_cleaned = df.copy()
        
        # Ensure date column is datetime
        df_cleaned[date_col] = pd.to_datetime(df_cleaned[date_col], errors='coerce')
        
        # Remove records with invalid dates
        invalid_dates = df_cleaned[date_col].isnull()
        if invalid_dates.any():
            logger.warning(f"Removing {invalid_dates.sum()} records with invalid dates")
            df_cleaned = df_cleaned[~invalid_dates]
        
        # Sort by patient ID and date
        df_cleaned = df_cleaned.sort_values([patient_id_col, date_col])
        
        # Filter patients with minimum observations
        min_obs = self.temporal_config.get('min_observations_per_patient', 2)
        patient_counts = df_cleaned[patient_id_col].value_counts()
        valid_patients = patient_counts[patient_counts >= min_obs].index
        
        df_cleaned = df_cleaned[df_cleaned[patient_id_col].isin(valid_patients)]
        
        logger.info(f"Temporal cleaning: {len(valid_patients)} patients with ≥{min_obs} observations")
        
        # Handle large gaps in temporal data
        max_gap_days = self.temporal_config.get('max_gap_days', 365)
        df_cleaned = self._handle_temporal_gaps(df_cleaned, patient_id_col, date_col, max_gap_days)
        
        return df_cleaned
    
    def _handle_temporal_gaps(self, df: pd.DataFrame, patient_id_col: str,
                            date_col: str, max_gap_days: int) -> pd.DataFrame:
        """
        Handle large gaps in temporal data by splitting patient records.
        
        Args:
            df: DataFrame with temporal data
            patient_id_col: Name of the patient ID column
            date_col: Name of the date column
            max_gap_days: Maximum allowed gap in days
            
        Returns:
            DataFrame with temporal gaps handled
        """
        df_gap_handled = df.copy()
        
        # Calculate gaps between consecutive observations for each patient
        df_gap_handled['date_diff'] = df_gap_handled.groupby(patient_id_col)[date_col].diff()
        df_gap_handled['gap_days'] = df_gap_handled['date_diff'].dt.days
        
        # Identify large gaps
        large_gaps = df_gap_handled['gap_days'] > max_gap_days
        
        if large_gaps.any():
            logger.warning(f"Found {large_gaps.sum()} large gaps (> {max_gap_days} days)")
            
            # For now, we'll keep all data but flag large gaps
            # In a production system, you might want to split patient records
            df_gap_handled['large_gap'] = large_gaps
        
        # Remove temporary columns
        df_gap_handled = df_gap_handled.drop(columns=['date_diff', 'gap_days'])
        
        return df_gap_handled
    
    def validate_clinical_ranges(self, df: pd.DataFrame,
                               clinical_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """
        Validate clinical values against expected ranges.
        
        Args:
            df: DataFrame containing clinical data
            clinical_ranges: Dictionary mapping column names to (min, max) ranges
            
        Returns:
            Dictionary containing validation results
        """
        logger.info("Validating clinical ranges")
        
        validation_results = {
            'valid_columns': [],
            'invalid_columns': [],
            'out_of_range_summary': {},
            'total_out_of_range': 0
        }
        
        for col, (min_val, max_val) in clinical_ranges.items():
            if col not in df.columns:
                logger.warning(f"Column {col} not found in data")
                continue
            
            # Check for values outside clinical range
            out_of_range = (df[col] < min_val) | (df[col] > max_val)
            out_of_range_count = out_of_range.sum()
            
            if out_of_range_count > 0:
                validation_results['invalid_columns'].append(col)
                validation_results['out_of_range_summary'][col] = {
                    'count': out_of_range_count,
                    'percentage': (out_of_range_count / len(df)) * 100,
                    'min_found': df[col].min(),
                    'max_found': df[col].max(),
                    'expected_range': (min_val, max_val)
                }
                validation_results['total_out_of_range'] += out_of_range_count
                
                logger.warning(f"Column {col}: {out_of_range_count} values outside range [{min_val}, {max_val}]")
            else:
                validation_results['valid_columns'].append(col)
        
        logger.info(f"Clinical range validation completed. "
                   f"{len(validation_results['valid_columns'])} valid columns, "
                   f"{len(validation_results['invalid_columns'])} invalid columns")
        
        return validation_results
    
    def generate_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive data quality report.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary containing data quality metrics
        """
        logger.info("Generating data quality report")
        
        quality_report = {
            'basic_stats': {
                'total_records': len(df),
                'total_columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
            },
            'missing_data': {},
            'duplicate_data': {
                'duplicate_records': df.duplicated().sum(),
                'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
            },
            'data_types': {},
            'numeric_summary': {},
            'categorical_summary': {}
        }
        
        # Missing data analysis
        missing_data = df.isnull().sum()
        quality_report['missing_data'] = {
            'total_missing': missing_data.sum(),
            'missing_percentage': (missing_data.sum() / (len(df) * len(df.columns))) * 100,
            'columns_with_missing': (missing_data > 0).sum(),
            'missing_by_column': missing_data[missing_data > 0].to_dict()
        }
        
        # Data type analysis
        for col in df.columns:
            quality_report['data_types'][col] = str(df[col].dtype)
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            quality_report['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            quality_report['categorical_summary'][col] = {
                'unique_values': df[col].nunique(),
                'most_common': df[col].value_counts().head(5).to_dict()
            }
        
        logger.info("Data quality report generated")
        return quality_report
