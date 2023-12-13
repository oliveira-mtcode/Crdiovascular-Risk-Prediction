"""
Data preprocessing module for cardiovascular risk prediction pipeline.

This module provides comprehensive data preprocessing capabilities including
feature engineering, temporal data processing, and data transformation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
import warnings

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles data preprocessing operations for clinical data.
    
    Provides methods for feature engineering, temporal data processing,
    and data transformation specific to cardiovascular risk prediction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data preprocessor with configuration.
        
        Args:
            config: Configuration dictionary containing preprocessing parameters
        """
        self.config = config
        self.feature_config = config.get('feature_engineering', {})
        self.temporal_config = self.feature_config.get('temporal_features', {})
        self.clinical_config = self.feature_config.get('clinical_features', {})
        
        # Initialize scalers and encoders
        self.scalers = {}
        self.encoders = {}
        self.imputer = None
        
        logger.info("Data preprocessor initialized")
    
    def engineer_temporal_features(self, df: pd.DataFrame,
                                 patient_id_col: str = 'patient_id',
                                 date_col: str = 'date') -> pd.DataFrame:
        """
        Engineer temporal features from longitudinal clinical data.
        
        Args:
            df: DataFrame containing longitudinal data
            patient_id_col: Name of the patient ID column
            date_col: Name of the date column
            
        Returns:
            DataFrame with engineered temporal features
        """
        logger.info("Engineering temporal features")
        
        df_engineered = df.copy()
        
        # Ensure date column is datetime
        df_engineered[date_col] = pd.to_datetime(df_engineered[date_col])
        
        # Sort by patient and date
        df_engineered = df_engineered.sort_values([patient_id_col, date_col])
        
        # Get temporal aggregation methods
        aggregations = self.temporal_config.get('aggregations', ['mean', 'std', 'min', 'max'])
        time_windows = self.temporal_config.get('time_windows', [30, 90, 180, 365])
        
        # Identify numeric columns for temporal feature engineering
        numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns.tolist()
        # Remove patient_id and date columns from numeric features
        numeric_cols = [col for col in numeric_cols if col not in [patient_id_col, date_col]]
        
        # Create temporal features for each numeric column
        for col in numeric_cols:
            if col in df_engineered.columns:
                df_engineered = self._create_temporal_features_for_column(
                    df_engineered, col, patient_id_col, date_col, aggregations, time_windows
                )
        
        logger.info(f"Temporal feature engineering completed for {len(numeric_cols)} columns")
        return df_engineered
    
    def _create_temporal_features_for_column(self, df: pd.DataFrame, column: str,
                                           patient_id_col: str, date_col: str,
                                           aggregations: List[str],
                                           time_windows: List[int]) -> pd.DataFrame:
        """
        Create temporal features for a specific column.
        
        Args:
            df: DataFrame containing the data
            column: Name of the column to create features for
            patient_id_col: Name of the patient ID column
            date_col: Name of the date column
            aggregations: List of aggregation methods
            time_windows: List of time windows in days
            
        Returns:
            DataFrame with temporal features added
        """
        df_result = df.copy()
        
        # Create rolling window features
        for window_days in time_windows:
            window_name = f"{window_days}d"
            
            # Calculate rolling statistics
            rolling_stats = df_result.groupby(patient_id_col)[column].rolling(
                window=f'{window_days}D', on=date_col, min_periods=1
            )
            
            for agg_method in aggregations:
                if agg_method == 'mean':
                    df_result[f'{column}_{window_name}_mean'] = rolling_stats.mean().values
                elif agg_method == 'std':
                    df_result[f'{column}_{window_name}_std'] = rolling_stats.std().values
                elif agg_method == 'min':
                    df_result[f'{column}_{window_name}_min'] = rolling_stats.min().values
                elif agg_method == 'max':
                    df_result[f'{column}_{window_name}_max'] = rolling_stats.max().values
                elif agg_method == 'trend':
                    df_result[f'{column}_{window_name}_trend'] = self._calculate_trend(
                        df_result, column, patient_id_col, date_col, window_days
                    )
                elif agg_method == 'variability':
                    df_result[f'{column}_{window_name}_variability'] = self._calculate_variability(
                        df_result, column, patient_id_col, date_col, window_days
                    )
        
        return df_result
    
    def _calculate_trend(self, df: pd.DataFrame, column: str, patient_id_col: str,
                        date_col: str, window_days: int) -> pd.Series:
        """
        Calculate trend (slope) for a column over a time window.
        
        Args:
            df: DataFrame containing the data
            column: Name of the column
            patient_id_col: Name of the patient ID column
            date_col: Name of the date column
            window_days: Time window in days
            
        Returns:
            Series containing trend values
        """
        trend_values = pd.Series([np.nan] * len(df), index=df.index)
        
        for patient_id in df[patient_id_col].unique():
            patient_data = df[df[patient_id_col] == patient_id].copy()
            patient_data = patient_data.sort_values(date_col)
            
            # Calculate rolling trend
            for i in range(len(patient_data)):
                end_date = patient_data.iloc[i][date_col]
                start_date = end_date - timedelta(days=window_days)
                
                window_data = patient_data[
                    (patient_data[date_col] >= start_date) & 
                    (patient_data[date_col] <= end_date)
                ]
                
                if len(window_data) >= 2:
                    # Calculate linear trend
                    x = np.arange(len(window_data))
                    y = window_data[column].values
                    
                    if not np.isnan(y).all():
                        # Remove NaN values
                        valid_mask = ~np.isnan(y)
                        if valid_mask.sum() >= 2:
                            x_valid = x[valid_mask]
                            y_valid = y[valid_mask]
                            
                            # Calculate slope
                            slope = np.polyfit(x_valid, y_valid, 1)[0]
                            trend_values.loc[window_data.index[-1]] = slope
        
        return trend_values
    
    def _calculate_variability(self, df: pd.DataFrame, column: str, patient_id_col: str,
                             date_col: str, window_days: int) -> pd.Series:
        """
        Calculate variability (coefficient of variation) for a column over a time window.
        
        Args:
            df: DataFrame containing the data
            column: Name of the column
            patient_id_col: Name of the patient ID column
            date_col: Name of the date column
            window_days: Time window in days
            
        Returns:
            Series containing variability values
        """
        variability_values = pd.Series([np.nan] * len(df), index=df.index)
        
        for patient_id in df[patient_id_col].unique():
            patient_data = df[df[patient_id_col] == patient_id].copy()
            patient_data = patient_data.sort_values(date_col)
            
            # Calculate rolling variability
            for i in range(len(patient_data)):
                end_date = patient_data.iloc[i][date_col]
                start_date = end_date - timedelta(days=window_days)
                
                window_data = patient_data[
                    (patient_data[date_col] >= start_date) & 
                    (patient_data[date_col] <= end_date)
                ]
                
                if len(window_data) >= 2:
                    values = window_data[column].dropna()
                    if len(values) >= 2 and values.mean() != 0:
                        cv = values.std() / values.mean()
                        variability_values.loc[window_data.index[-1]] = cv
        
        return variability_values
    
    def engineer_clinical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer clinical features for cardiovascular risk prediction.
        
        Args:
            df: DataFrame containing clinical data
            
        Returns:
            DataFrame with engineered clinical features
        """
        logger.info("Engineering clinical features")
        
        df_engineered = df.copy()
        
        # Create derived clinical features
        df_engineered = self._create_bmi_features(df_engineered)
        df_engineered = self._create_bp_features(df_engineered)
        df_engineered = self._create_lipid_features(df_engineered)
        df_engineered = self._create_risk_scores(df_engineered)
        df_engineered = self._create_age_features(df_engineered)
        
        logger.info("Clinical feature engineering completed")
        return df_engineered
    
    def _create_bmi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create BMI-related features."""
        df_result = df.copy()
        
        # Calculate BMI if height and weight are available
        if 'height' in df.columns and 'weight' in df.columns:
            # Convert height to meters if needed
            height_m = df_result['height']
            if height_m.max() > 3:  # Assume height is in cm if max > 3
                height_m = height_m / 100
            
            df_result['bmi'] = df_result['weight'] / (height_m ** 2)
            
            # BMI categories
            df_result['bmi_category'] = pd.cut(
                df_result['bmi'],
                bins=[0, 18.5, 25, 30, 35, 100],
                labels=['Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II+']
            )
        
        return df_result
    
    def _create_bp_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create blood pressure-related features."""
        df_result = df.copy()
        
        if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
            # Mean arterial pressure
            df_result['map'] = (2 * df_result['diastolic_bp'] + df_result['systolic_bp']) / 3
            
            # Pulse pressure
            df_result['pulse_pressure'] = df_result['systolic_bp'] - df_result['diastolic_bp']
            
            # BP categories
            df_result['bp_category'] = pd.cut(
                df_result['systolic_bp'],
                bins=[0, 120, 130, 140, 180, 300],
                labels=['Normal', 'Elevated', 'Stage 1', 'Stage 2', 'Crisis']
            )
        
        return df_result
    
    def _create_lipid_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lipid-related features."""
        df_result = df.copy()
        
        # Non-HDL cholesterol
        if 'total_cholesterol' in df.columns and 'hdl_cholesterol' in df.columns:
            df_result['non_hdl_cholesterol'] = df_result['total_cholesterol'] - df_result['hdl_cholesterol']
        
        # Total cholesterol to HDL ratio
        if 'total_cholesterol' in df.columns and 'hdl_cholesterol' in df.columns:
            df_result['tc_hdl_ratio'] = df_result['total_cholesterol'] / df_result['hdl_cholesterol']
        
        # LDL to HDL ratio
        if 'ldl_cholesterol' in df.columns and 'hdl_cholesterol' in df.columns:
            df_result['ldl_hdl_ratio'] = df_result['ldl_cholesterol'] / df_result['hdl_cholesterol']
        
        # Triglycerides to HDL ratio
        if 'triglycerides' in df.columns and 'hdl_cholesterol' in df.columns:
            df_result['tg_hdl_ratio'] = df_result['triglycerides'] / df_result['hdl_cholesterol']
        
        return df_result
    
    def _create_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cardiovascular risk scores."""
        df_result = df.copy()
        
        # Simple risk score based on available factors
        risk_score = 0
        
        # Age factor
        if 'age' in df.columns:
            age_risk = np.where(df_result['age'] >= 65, 2, 
                              np.where(df_result['age'] >= 45, 1, 0))
            risk_score += age_risk
        
        # Sex factor
        if 'sex' in df.columns:
            sex_risk = np.where(df_result['sex'].str.lower().isin(['male', 'm', '1']), 1, 0)
            risk_score += sex_risk
        
        # Smoking factor
        if 'smoking_status' in df.columns:
            smoking_risk = np.where(df_result['smoking_status'].str.lower().isin(['yes', 'current', '1']), 2, 0)
            risk_score += smoking_risk
        
        # Diabetes factor
        if 'diabetes_status' in df.columns:
            diabetes_risk = np.where(df_result['diabetes_status'].str.lower().isin(['yes', '1']), 2, 0)
            risk_score += diabetes_risk
        
        # High BP factor
        if 'systolic_bp' in df.columns:
            bp_risk = np.where(df_result['systolic_bp'] >= 140, 1, 0)
            risk_score += bp_risk
        
        # High cholesterol factor
        if 'total_cholesterol' in df.columns:
            chol_risk = np.where(df_result['total_cholesterol'] >= 240, 1, 0)
            risk_score += chol_risk
        
        df_result['simple_risk_score'] = risk_score
        
        # Risk categories
        df_result['risk_category'] = pd.cut(
            df_result['simple_risk_score'],
            bins=[-1, 2, 4, 6, 10],
            labels=['Low', 'Moderate', 'High', 'Very High']
        )
        
        return df_result
    
    def _create_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create age-related features."""
        df_result = df.copy()
        
        if 'age' in df.columns:
            # Age groups
            df_result['age_group'] = pd.cut(
                df_result['age'],
                bins=[0, 40, 60, 80, 100],
                labels=['Young', 'Middle-aged', 'Elderly', 'Very Elderly']
            )
            
            # Age squared (for non-linear relationships)
            df_result['age_squared'] = df_result['age'] ** 2
        
        return df_result
    
    def encode_categorical_features(self, df: pd.DataFrame,
                                  categorical_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Encode categorical features for machine learning.
        
        Args:
            df: DataFrame containing the data
            categorical_columns: List of categorical columns to encode
            
        Returns:
            DataFrame with encoded categorical features
        """
        logger.info("Encoding categorical features")
        
        df_encoded = df.copy()
        
        if categorical_columns is None:
            categorical_columns = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                # Use label encoding for ordinal categories
                if col in ['age_group', 'bmi_category', 'bp_category', 'risk_category']:
                    le = LabelEncoder()
                    df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].astype(str))
                    self.encoders[col] = le
                else:
                    # Use one-hot encoding for nominal categories
                    dummies = pd.get_dummies(df_encoded[col], prefix=col, dummy_na=True)
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
                    df_encoded = df_encoded.drop(columns=[col])
        
        logger.info(f"Encoded {len(categorical_columns)} categorical features")
        return df_encoded
    
    def scale_numeric_features(self, df: pd.DataFrame,
                             numeric_columns: Optional[List[str]] = None,
                             fit_scaler: bool = True) -> pd.DataFrame:
        """
        Scale numeric features for machine learning.
        
        Args:
            df: DataFrame containing the data
            numeric_columns: List of numeric columns to scale
            fit_scaler: Whether to fit the scaler (True for training, False for testing)
            
        Returns:
            DataFrame with scaled numeric features
        """
        logger.info("Scaling numeric features")
        
        df_scaled = df.copy()
        
        if numeric_columns is None:
            numeric_columns = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_columns:
            if col in df_scaled.columns:
                if fit_scaler:
                    scaler = StandardScaler()
                    df_scaled[col] = scaler.fit_transform(df_scaled[[col]])
                    self.scalers[col] = scaler
                else:
                    if col in self.scalers:
                        df_scaled[col] = self.scalers[col].transform(df_scaled[[col]])
                    else:
                        logger.warning(f"Scaler not found for column {col}")
        
        logger.info(f"Scaled {len(numeric_columns)} numeric features")
        return df_scaled
    
    def handle_missing_values_advanced(self, df: pd.DataFrame,
                                     numeric_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Handle missing values using advanced imputation methods.
        
        Args:
            df: DataFrame containing the data
            numeric_columns: List of numeric columns to impute
            
        Returns:
            DataFrame with imputed values
        """
        logger.info("Handling missing values with advanced imputation")
        
        df_imputed = df.copy()
        
        if numeric_columns is None:
            numeric_columns = df_imputed.select_dtypes(include=[np.number]).columns.tolist()
        
        # Use KNN imputation for numeric columns
        if numeric_columns:
            if self.imputer is None:
                self.imputer = KNNImputer(n_neighbors=5)
                df_imputed[numeric_columns] = self.imputer.fit_transform(df_imputed[numeric_columns])
            else:
                df_imputed[numeric_columns] = self.imputer.transform(df_imputed[numeric_columns])
        
        logger.info("Advanced missing value imputation completed")
        return df_imputed
    
    def create_feature_matrix(self, df: pd.DataFrame,
                            target_column: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Create feature matrix for machine learning.
        
        Args:
            df: DataFrame containing the data
            target_column: Name of the target column (optional)
            
        Returns:
            Tuple of (feature_matrix, target_vector)
        """
        logger.info("Creating feature matrix")
        
        # Select numeric columns for feature matrix
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column if specified
        if target_column and target_column in feature_columns:
            feature_columns.remove(target_column)
        
        # Create feature matrix
        X = df[feature_columns].copy()
        
        # Create target vector if specified
        y = None
        if target_column and target_column in df.columns:
            y = df[target_column].copy()
        
        logger.info(f"Feature matrix created: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y


