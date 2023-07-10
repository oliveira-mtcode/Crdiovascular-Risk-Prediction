"""
Validation utilities for cardiovascular risk prediction pipeline.

This module provides comprehensive validation capabilities for data
and model validation in the cardiovascular risk prediction pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Provides data validation capabilities for clinical data.
    
    Validates data quality, structure, and clinical ranges
    for cardiovascular risk prediction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data validator with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.clinical_ranges = self._get_clinical_ranges()
        
        logger.info("Data validator initialized")
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate overall data quality.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary containing validation results
        """
        logger.info("Validating data quality")
        
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'quality_score': 0.0,
            'summary': {}
        }
        
        # Check basic data structure
        structure_validation = self._validate_data_structure(df)
        validation_results['summary']['structure'] = structure_validation
        
        if not structure_validation['is_valid']:
            validation_results['is_valid'] = False
            validation_results['issues'].extend(structure_validation['issues'])
        
        # Check data completeness
        completeness_validation = self._validate_data_completeness(df)
        validation_results['summary']['completeness'] = completeness_validation
        
        if completeness_validation['completeness_score'] < 0.7:
            validation_results['warnings'].append("Low data completeness")
        
        # Check clinical ranges
        range_validation = self._validate_clinical_ranges(df)
        validation_results['summary']['clinical_ranges'] = range_validation
        
        if range_validation['out_of_range_count'] > 0:
            validation_results['warnings'].append("Values outside clinical ranges detected")
        
        # Check data consistency
        consistency_validation = self._validate_data_consistency(df)
        validation_results['summary']['consistency'] = consistency_validation
        
        if not consistency_validation['is_consistent']:
            validation_results['warnings'].append("Data consistency issues detected")
        
        # Calculate overall quality score
        validation_results['quality_score'] = self._calculate_quality_score(validation_results)
        
        logger.info(f"Data quality validation completed. Score: {validation_results['quality_score']:.2f}")
        return validation_results
    
    def validate_metabolic_syndrome_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data for metabolic syndrome identification.
        
        Args:
            df: DataFrame containing clinical data
            
        Returns:
            Dictionary containing validation results
        """
        logger.info("Validating metabolic syndrome data")
        
        validation_results = {
            'is_valid': True,
            'missing_columns': [],
            'data_quality_issues': [],
            'validation_passed': True
        }
        
        # Required columns for metabolic syndrome identification
        required_columns = [
            'waist_circumference', 'systolic_bp', 'diastolic_bp',
            'glucose', 'triglycerides', 'hdl_cholesterol'
        ]
        
        # Check for required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['missing_columns'] = missing_cols
            validation_results['is_valid'] = False
            validation_results['validation_passed'] = False
        
        # Check for missing values in critical columns
        for col in required_columns:
            if col in df.columns:
                missing_pct = (df[col].isnull().sum() / len(df)) * 100
                if missing_pct > 50:
                    validation_results['data_quality_issues'].append(
                        f"High missing values in {col}: {missing_pct:.1f}%"
                    )
        
        # Check for sex column (needed for thresholds)
        if 'sex' not in df.columns:
            validation_results['data_quality_issues'].append("Sex column missing - using default thresholds")
        
        logger.info(f"Metabolic syndrome data validation: {'PASSED' if validation_results['validation_passed'] else 'FAILED'}")
        return validation_results
    
    def validate_temporal_data(self, df: pd.DataFrame, 
                             patient_id_col: str = 'patient_id',
                             date_col: str = 'date') -> Dict[str, Any]:
        """
        Validate temporal data structure and quality.
        
        Args:
            df: DataFrame containing temporal data
            patient_id_col: Name of the patient ID column
            date_col: Name of the date column
            
        Returns:
            Dictionary containing validation results
        """
        logger.info("Validating temporal data")
        
        validation_results = {
            'is_valid': True,
            'issues': [],
            'temporal_quality_score': 0.0,
            'summary': {}
        }
        
        # Check for required columns
        if patient_id_col not in df.columns:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Patient ID column '{patient_id_col}' not found")
        
        if date_col not in df.columns:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Date column '{date_col}' not found")
        
        if not validation_results['is_valid']:
            return validation_results
        
        # Check date column format
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Invalid date format: {e}")
            return validation_results
        
        # Analyze temporal patterns
        temporal_analysis = self._analyze_temporal_patterns(df, patient_id_col, date_col)
        validation_results['summary']['temporal_analysis'] = temporal_analysis
        
        # Check for temporal gaps
        gap_analysis = self._analyze_temporal_gaps(df, patient_id_col, date_col)
        validation_results['summary']['gap_analysis'] = gap_analysis
        
        # Calculate temporal quality score
        validation_results['temporal_quality_score'] = self._calculate_temporal_quality_score(validation_results)
        
        logger.info(f"Temporal data validation completed. Score: {validation_results['temporal_quality_score']:.2f}")
        return validation_results
    
    def _validate_data_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate basic data structure."""
        return {
            'is_valid': len(df) > 0,
            'total_records': len(df),
            'total_columns': len(df.columns),
            'issues': [] if len(df) > 0 else ['Empty dataset']
        }
    
    def _validate_data_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data completeness."""
        missing_data = df.isnull().sum()
        total_cells = len(df) * len(df.columns)
        missing_cells = missing_data.sum()
        
        completeness_score = 1 - (missing_cells / total_cells)
        
        return {
            'completeness_score': completeness_score,
            'missing_cells': int(missing_cells),
            'total_cells': int(total_cells),
            'missing_percentage': (missing_cells / total_cells) * 100
        }
    
    def _validate_clinical_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate clinical value ranges."""
        out_of_range_count = 0
        range_violations = {}
        
        for column, (min_val, max_val) in self.clinical_ranges.items():
            if column in df.columns:
                out_of_range = (df[column] < min_val) | (df[column] > max_val)
                count = out_of_range.sum()
                if count > 0:
                    range_violations[column] = {
                        'count': int(count),
                        'percentage': (count / len(df)) * 100,
                        'range': [min_val, max_val]
                    }
                    out_of_range_count += count
        
        return {
            'out_of_range_count': int(out_of_range_count),
            'range_violations': range_violations
        }
    
    def _validate_data_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data consistency."""
        consistency_issues = []
        
        # Check for duplicate records
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            consistency_issues.append(f"{duplicates} duplicate records found")
        
        # Check for logical inconsistencies
        if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
            invalid_bp = df['systolic_bp'] < df['diastolic_bp']
            if invalid_bp.any():
                consistency_issues.append(f"{invalid_bp.sum()} records with systolic < diastolic BP")
        
        return {
            'is_consistent': len(consistency_issues) == 0,
            'consistency_issues': consistency_issues
        }
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame, 
                                 patient_id_col: str, date_col: str) -> Dict[str, Any]:
        """Analyze temporal data patterns."""
        # Calculate time span
        time_span = (df[date_col].max() - df[date_col].min()).days
        
        # Calculate observations per patient
        patient_counts = df[patient_id_col].value_counts()
        
        return {
            'time_span_days': time_span,
            'unique_patients': len(patient_counts),
            'total_observations': len(df),
            'observations_per_patient': {
                'mean': float(patient_counts.mean()),
                'std': float(patient_counts.std()),
                'min': int(patient_counts.min()),
                'max': int(patient_counts.max())
            }
        }
    
    def _analyze_temporal_gaps(self, df: pd.DataFrame, 
                             patient_id_col: str, date_col: str) -> Dict[str, Any]:
        """Analyze temporal gaps in data."""
        gaps = []
        
        for patient_id in df[patient_id_col].unique():
            patient_data = df[df[patient_id_col] == patient_id].sort_values(date_col)
            if len(patient_data) > 1:
                time_diffs = patient_data[date_col].diff().dt.days.dropna()
                gaps.extend(time_diffs.tolist())
        
        if gaps:
            return {
                'mean_gap_days': float(np.mean(gaps)),
                'median_gap_days': float(np.median(gaps)),
                'max_gap_days': float(np.max(gaps)),
                'gap_count': len(gaps)
            }
        else:
            return {
                'mean_gap_days': 0,
                'median_gap_days': 0,
                'max_gap_days': 0,
                'gap_count': 0
            }
    
    def _calculate_quality_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall data quality score."""
        score = 1.0
        
        # Deduct for structure issues
        if not validation_results['summary']['structure']['is_valid']:
            score -= 0.3
        
        # Deduct for completeness issues
        completeness_score = validation_results['summary']['completeness']['completeness_score']
        score *= completeness_score
        
        # Deduct for range violations
        range_violations = validation_results['summary']['clinical_ranges']['range_violations']
        if range_violations:
            score -= 0.1
        
        # Deduct for consistency issues
        if not validation_results['summary']['consistency']['is_consistent']:
            score -= 0.1
        
        return max(0.0, score)
    
    def _calculate_temporal_quality_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate temporal data quality score."""
        score = 1.0
        
        # Check for temporal issues
        temporal_analysis = validation_results['summary']['temporal_analysis']
        gap_analysis = validation_results['summary']['gap_analysis']
        
        # Deduct for large gaps
        if gap_analysis['max_gap_days'] > 365:
            score -= 0.2
        
        # Deduct for low observation density
        obs_per_patient = temporal_analysis['observations_per_patient']['mean']
        if obs_per_patient < 2:
            score -= 0.3
        
        return max(0.0, score)
    
    def _get_clinical_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Get clinical value ranges for validation."""
        return {
            'age': (0, 120),
            'systolic_bp': (70, 250),
            'diastolic_bp': (40, 150),
            'total_cholesterol': (100, 500),
            'ldl_cholesterol': (50, 300),
            'hdl_cholesterol': (10, 150),
            'triglycerides': (50, 1000),
            'glucose': (50, 500),
            'hba1c': (3, 15),
            'bmi': (10, 80),
            'waist_circumference': (50, 200)
        }


class ModelValidator:
    """
    Provides model validation capabilities.
    
    Validates model performance, calibration, and clinical utility
    for cardiovascular risk prediction models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model validator with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.validation_config = config.get('validation', {})
        
        logger.info("Model validator initialized")
    
    def validate_model_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_scores: np.ndarray) -> Dict[str, Any]:
        """
        Validate model performance against clinical requirements.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_scores: Prediction scores/probabilities
            
        Returns:
            Dictionary containing validation results
        """
        logger.info("Validating model performance")
        
        validation_results = {
            'is_valid': True,
            'performance_requirements_met': True,
            'issues': [],
            'warnings': [],
            'performance_summary': {}
        }
        
        # Calculate performance metrics
        from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
        
        try:
            auc_roc = roc_auc_score(y_true, y_scores)
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            
            validation_results['performance_summary'] = {
                'auc_roc': float(auc_roc),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall)
            }
            
            # Check performance requirements
            if auc_roc < 0.7:
                validation_results['performance_requirements_met'] = False
                validation_results['issues'].append(f"AUC-ROC below threshold: {auc_roc:.3f} < 0.7")
            
            if recall < 0.7:
                validation_results['warnings'].append(f"Low sensitivity: {recall:.3f}")
            
            if precision < 0.6:
                validation_results['warnings'].append(f"Low precision: {precision:.3f}")
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Error calculating performance metrics: {e}")
        
        logger.info(f"Model performance validation: {'PASSED' if validation_results['performance_requirements_met'] else 'FAILED'}")
        return validation_results
    
    def validate_model_calibration(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, Any]:
        """
        Validate model calibration.
        
        Args:
            y_true: True binary labels
            y_scores: Prediction scores/probabilities
            
        Returns:
            Dictionary containing calibration validation results
        """
        logger.info("Validating model calibration")
        
        validation_results = {
            'is_well_calibrated': True,
            'calibration_issues': [],
            'calibration_summary': {}
        }
        
        try:
            from sklearn.calibration import calibration_curve
            from sklearn.metrics import brier_score_loss
            
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_scores, n_bins=10, strategy='quantile'
            )
            
            # Calculate calibration slope
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression()
            reg.fit(mean_predicted_value.reshape(-1, 1), fraction_of_positives)
            calibration_slope = reg.coef_[0]
            calibration_intercept = reg.intercept_
            
            # Calculate Brier score
            brier_score = brier_score_loss(y_true, y_scores)
            
            validation_results['calibration_summary'] = {
                'calibration_slope': float(calibration_slope),
                'calibration_intercept': float(calibration_intercept),
                'brier_score': float(brier_score)
            }
            
            # Check calibration requirements
            if not (0.8 <= calibration_slope <= 1.2):
                validation_results['is_well_calibrated'] = False
                validation_results['calibration_issues'].append(
                    f"Calibration slope outside acceptable range: {calibration_slope:.3f}"
                )
            
            if brier_score > 0.25:
                validation_results['calibration_issues'].append(
                    f"High Brier score: {brier_score:.3f}"
                )
            
        except Exception as e:
            validation_results['is_well_calibrated'] = False
            validation_results['calibration_issues'].append(f"Error in calibration analysis: {e}")
        
        logger.info(f"Model calibration validation: {'PASSED' if validation_results['is_well_calibrated'] else 'FAILED'}")
        return validation_results
    
    def validate_clinical_utility(self, y_true: np.ndarray, y_scores: np.ndarray,
                                prevalence: float) -> Dict[str, Any]:
        """
        Validate clinical utility of the model.
        
        Args:
            y_true: True binary labels
            y_scores: Prediction scores/probabilities
            prevalence: Disease prevalence in the population
            
        Returns:
            Dictionary containing clinical utility validation results
        """
        logger.info("Validating clinical utility")
        
        validation_results = {
            'is_clinically_useful': True,
            'utility_issues': [],
            'clinical_summary': {}
        }
        
        try:
            from sklearn.metrics import roc_auc_score, precision_score, recall_score
            
            # Calculate performance metrics
            auc_roc = roc_auc_score(y_true, y_scores)
            
            # Calculate PPV and NPV at different thresholds
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
            ppv_npv_results = {}
            
            for threshold in thresholds:
                y_pred_thresh = (y_scores >= threshold).astype(int)
                precision = precision_score(y_true, y_pred_thresh, zero_division=0)
                recall = recall_score(y_true, y_pred_thresh, zero_division=0)
                
                # Calculate NPV
                tn = ((y_true == 0) & (y_pred_thresh == 0)).sum()
                fn = ((y_true == 1) & (y_pred_thresh == 0)).sum()
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                
                ppv_npv_results[threshold] = {
                    'ppv': float(precision),
                    'npv': float(npv),
                    'sensitivity': float(recall)
                }
            
            validation_results['clinical_summary'] = {
                'auc_roc': float(auc_roc),
                'prevalence': float(prevalence),
                'threshold_analysis': ppv_npv_results
            }
            
            # Check clinical utility requirements
            if auc_roc < 0.75:
                validation_results['is_clinically_useful'] = False
                validation_results['utility_issues'].append(
                    f"AUC-ROC below clinical utility threshold: {auc_roc:.3f} < 0.75"
                )
            
            # Check if model provides meaningful risk stratification
            high_risk_threshold = 0.2
            high_risk_patients = (y_scores >= high_risk_threshold).sum()
            high_risk_prevalence = high_risk_patients / len(y_scores)
            
            if high_risk_prevalence < 0.1 or high_risk_prevalence > 0.5:
                validation_results['utility_issues'].append(
                    f"High-risk prevalence outside optimal range: {high_risk_prevalence:.3f}"
                )
            
        except Exception as e:
            validation_results['is_clinically_useful'] = False
            validation_results['utility_issues'].append(f"Error in clinical utility analysis: {e}")
        
        logger.info(f"Clinical utility validation: {'PASSED' if validation_results['is_clinically_useful'] else 'FAILED'}")
        return validation_results

