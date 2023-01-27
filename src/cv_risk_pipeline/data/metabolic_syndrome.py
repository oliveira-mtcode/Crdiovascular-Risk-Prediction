"""
Metabolic syndrome identification module for cardiovascular risk prediction pipeline.

This module implements the identification of metabolic syndrome based on
ATP III guidelines and other clinical criteria.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class MetabolicSyndromeCriteria(Enum):
    """Enumeration of metabolic syndrome criteria."""
    WAIST_CIRCUMFERENCE = "waist_circumference"
    BLOOD_PRESSURE = "blood_pressure"
    FASTING_GLUCOSE = "fasting_glucose"
    TRIGLYCERIDES = "triglycerides"
    HDL_CHOLESTEROL = "hdl_cholesterol"


class MetabolicSyndromeIdentifier:
    """
    Identifies patients with metabolic syndrome based on clinical criteria.
    
    Implements ATP III guidelines and other established criteria for
    metabolic syndrome identification.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize metabolic syndrome identifier with configuration.
        
        Args:
            config: Configuration dictionary containing criteria thresholds
        """
        self.config = config
        self.metabolic_config = config.get('metabolic_syndrome', {})
        self.criteria_config = self.metabolic_config.get('criteria', {})
        self.required_criteria = self.metabolic_config.get('required_criteria', 3)
        
        logger.info("Metabolic syndrome identifier initialized")
    
    def identify_metabolic_syndrome(self, df: pd.DataFrame,
                                  patient_id_col: str = 'patient_id',
                                  ethnicity_col: Optional[str] = None) -> pd.DataFrame:
        """
        Identify patients with metabolic syndrome.
        
        Args:
            df: DataFrame containing clinical data
            patient_id_col: Name of the patient ID column
            ethnicity_col: Name of the ethnicity column (optional)
            
        Returns:
            DataFrame with metabolic syndrome identification results
        """
        logger.info("Identifying metabolic syndrome patients")
        
        df_result = df.copy()
        
        # Check each criterion
        criteria_results = {}
        criteria_results['waist_circumference'] = self._check_waist_circumference(
            df_result, ethnicity_col
        )
        criteria_results['blood_pressure'] = self._check_blood_pressure(df_result)
        criteria_results['fasting_glucose'] = self._check_fasting_glucose(df_result)
        criteria_results['triglycerides'] = self._check_triglycerides(df_result)
        criteria_results['hdl_cholesterol'] = self._check_hdl_cholesterol(df_result)
        
        # Add individual criterion results to dataframe
        for criterion, result in criteria_results.items():
            df_result[f'ms_{criterion}_positive'] = result
        
        # Calculate total positive criteria
        criterion_columns = [f'ms_{criterion}_positive' for criterion in criteria_results.keys()]
        df_result['ms_criteria_count'] = df_result[criterion_columns].sum(axis=1)
        
        # Determine metabolic syndrome status
        df_result['metabolic_syndrome'] = df_result['ms_criteria_count'] >= self.required_criteria
        
        # Calculate metabolic syndrome severity
        df_result['ms_severity'] = self._calculate_severity(df_result['ms_criteria_count'])
        
        # Add summary statistics
        ms_count = df_result['metabolic_syndrome'].sum()
        ms_percentage = (ms_count / len(df_result)) * 100
        
        logger.info(f"Metabolic syndrome identification completed: "
                   f"{ms_count} patients ({ms_percentage:.1f}%) identified")
        
        return df_result
    
    def _check_waist_circumference(self, df: pd.DataFrame,
                                 ethnicity_col: Optional[str] = None) -> pd.Series:
        """
        Check waist circumference criterion for metabolic syndrome.
        
        Args:
            df: DataFrame containing clinical data
            ethnicity_col: Name of the ethnicity column
            
        Returns:
            Boolean Series indicating positive criterion
        """
        waist_col = 'waist_circumference'
        if waist_col not in df.columns:
            logger.warning(f"Waist circumference column '{waist_col}' not found")
            return pd.Series([False] * len(df), index=df.index)
        
        # Get thresholds from config
        waist_thresholds = self.criteria_config.get('waist_circumference', {})
        
        # Default thresholds (Caucasian)
        caucasian_male_threshold = waist_thresholds.get('caucasian_male', 102)
        caucasian_female_threshold = waist_thresholds.get('caucasian_female', 88)
        asian_male_threshold = waist_thresholds.get('asian_male', 90)
        asian_female_threshold = waist_thresholds.get('asian_female', 80)
        
        result = pd.Series([False] * len(df), index=df.index)
        
        # Check if sex column exists
        sex_col = 'sex'
        if sex_col not in df.columns:
            logger.warning(f"Sex column '{sex_col}' not found, using default thresholds")
            # Use Caucasian thresholds as default
            result = (df[waist_col] >= caucasian_male_threshold)
            return result
        
        # Apply ethnicity-specific thresholds if available
        if ethnicity_col and ethnicity_col in df.columns:
            # Asian patients
            asian_mask = df[ethnicity_col].str.lower().isin(['asian', 'east asian', 'south asian'])
            male_mask = df[sex_col].str.lower().isin(['male', 'm', '1'])
            female_mask = df[sex_col].str.lower().isin(['female', 'f', '0'])
            
            # Asian males
            asian_male_mask = asian_mask & male_mask
            result[asian_male_mask] = df.loc[asian_male_mask, waist_col] >= asian_male_threshold
            
            # Asian females
            asian_female_mask = asian_mask & female_mask
            result[asian_female_mask] = df.loc[asian_female_mask, waist_col] >= asian_female_threshold
            
            # Non-Asian patients (use Caucasian thresholds)
            non_asian_mask = ~asian_mask
            non_asian_male_mask = non_asian_mask & male_mask
            non_asian_female_mask = non_asian_mask & female_mask
            
            result[non_asian_male_mask] = df.loc[non_asian_male_mask, waist_col] >= caucasian_male_threshold
            result[non_asian_female_mask] = df.loc[non_asian_female_mask, waist_col] >= caucasian_female_threshold
        else:
            # Use sex-based thresholds (Caucasian)
            male_mask = df[sex_col].str.lower().isin(['male', 'm', '1'])
            female_mask = df[sex_col].str.lower().isin(['female', 'f', '0'])
            
            result[male_mask] = df.loc[male_mask, waist_col] >= caucasian_male_threshold
            result[female_mask] = df.loc[female_mask, waist_col] >= caucasian_female_threshold
        
        return result
    
    def _check_blood_pressure(self, df: pd.DataFrame) -> pd.Series:
        """
        Check blood pressure criterion for metabolic syndrome.
        
        Args:
            df: DataFrame containing clinical data
            
        Returns:
            Boolean Series indicating positive criterion
        """
        bp_config = self.criteria_config.get('blood_pressure', {})
        systolic_threshold = bp_config.get('systolic', 130)
        diastolic_threshold = bp_config.get('diastolic', 85)
        
        systolic_col = 'systolic_bp'
        diastolic_col = 'diastolic_bp'
        
        result = pd.Series([False] * len(df), index=df.index)
        
        # Check if both BP columns exist
        if systolic_col in df.columns and diastolic_col in df.columns:
            result = (df[systolic_col] >= systolic_threshold) | (df[diastolic_col] >= diastolic_threshold)
        elif systolic_col in df.columns:
            result = df[systolic_col] >= systolic_threshold
            logger.warning(f"Only systolic BP column found, using systolic threshold only")
        elif diastolic_col in df.columns:
            result = df[diastolic_col] >= diastolic_threshold
            logger.warning(f"Only diastolic BP column found, using diastolic threshold only")
        else:
            logger.warning(f"Blood pressure columns not found: {systolic_col}, {diastolic_col}")
        
        return result
    
    def _check_fasting_glucose(self, df: pd.DataFrame) -> pd.Series:
        """
        Check fasting glucose criterion for metabolic syndrome.
        
        Args:
            df: DataFrame containing clinical data
            
        Returns:
            Boolean Series indicating positive criterion
        """
        glucose_threshold = self.criteria_config.get('fasting_glucose', 100)
        glucose_col = 'glucose'
        
        if glucose_col not in df.columns:
            logger.warning(f"Glucose column '{glucose_col}' not found")
            return pd.Series([False] * len(df), index=df.index)
        
        result = df[glucose_col] >= glucose_threshold
        return result
    
    def _check_triglycerides(self, df: pd.DataFrame) -> pd.Series:
        """
        Check triglycerides criterion for metabolic syndrome.
        
        Args:
            df: DataFrame containing clinical data
            
        Returns:
            Boolean Series indicating positive criterion
        """
        triglycerides_threshold = self.criteria_config.get('triglycerides', 150)
        triglycerides_col = 'triglycerides'
        
        if triglycerides_col not in df.columns:
            logger.warning(f"Triglycerides column '{triglycerides_col}' not found")
            return pd.Series([False] * len(df), index=df.index)
        
        result = df[triglycerides_col] >= triglycerides_threshold
        return result
    
    def _check_hdl_cholesterol(self, df: pd.DataFrame) -> pd.Series:
        """
        Check HDL cholesterol criterion for metabolic syndrome.
        
        Args:
            df: DataFrame containing clinical data
            
        Returns:
            Boolean Series indicating positive criterion
        """
        hdl_config = self.criteria_config.get('hdl_cholesterol', {})
        male_threshold = hdl_config.get('male', 40)
        female_threshold = hdl_config.get('female', 50)
        
        hdl_col = 'hdl_cholesterol'
        sex_col = 'sex'
        
        if hdl_col not in df.columns:
            logger.warning(f"HDL cholesterol column '{hdl_col}' not found")
            return pd.Series([False] * len(df), index=df.index)
        
        result = pd.Series([False] * len(df), index=df.index)
        
        if sex_col in df.columns:
            male_mask = df[sex_col].str.lower().isin(['male', 'm', '1'])
            female_mask = df[sex_col].str.lower().isin(['female', 'f', '0'])
            
            # Low HDL is positive criterion (below threshold)
            result[male_mask] = df.loc[male_mask, hdl_col] < male_threshold
            result[female_mask] = df.loc[female_mask, hdl_col] < female_threshold
        else:
            # Use male threshold as default
            logger.warning(f"Sex column '{sex_col}' not found, using male HDL threshold")
            result = df[hdl_col] < male_threshold
        
        return result
    
    def _calculate_severity(self, criteria_count: pd.Series) -> pd.Series:
        """
        Calculate metabolic syndrome severity based on number of positive criteria.
        
        Args:
            criteria_count: Series containing number of positive criteria
            
        Returns:
            Series containing severity levels
        """
        severity = pd.Series(['None'] * len(criteria_count), index=criteria_count.index)
        
        severity[criteria_count == 3] = 'Mild'
        severity[criteria_count == 4] = 'Moderate'
        severity[criteria_count == 5] = 'Severe'
        
        return severity
    
    def get_metabolic_syndrome_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for metabolic syndrome identification.
        
        Args:
            df: DataFrame with metabolic syndrome identification results
            
        Returns:
            Dictionary containing summary statistics
        """
        if 'metabolic_syndrome' not in df.columns:
            raise ValueError("DataFrame must contain 'metabolic_syndrome' column")
        
        summary = {
            'total_patients': len(df),
            'metabolic_syndrome_patients': df['metabolic_syndrome'].sum(),
            'metabolic_syndrome_percentage': (df['metabolic_syndrome'].sum() / len(df)) * 100,
            'criteria_distribution': {},
            'severity_distribution': {}
        }
        
        # Criteria distribution
        if 'ms_criteria_count' in df.columns:
            criteria_dist = df['ms_criteria_count'].value_counts().sort_index()
            summary['criteria_distribution'] = criteria_dist.to_dict()
        
        # Severity distribution
        if 'ms_severity' in df.columns:
            severity_dist = df['ms_severity'].value_counts()
            summary['severity_distribution'] = severity_dist.to_dict()
        
        # Individual criteria prevalence
        criterion_columns = [col for col in df.columns if col.startswith('ms_') and col.endswith('_positive')]
        for col in criterion_columns:
            criterion_name = col.replace('ms_', '').replace('_positive', '')
            summary[f'{criterion_name}_prevalence'] = {
                'count': df[col].sum(),
                'percentage': (df[col].sum() / len(df)) * 100
            }
        
        logger.info(f"Metabolic syndrome summary: {summary['metabolic_syndrome_patients']} patients "
                   f"({summary['metabolic_syndrome_percentage']:.1f}%) identified")
        
        return summary
    
    def validate_metabolic_syndrome_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality for metabolic syndrome identification.
        
        Args:
            df: DataFrame containing clinical data
            
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'required_columns_present': True,
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
            validation_results['required_columns_present'] = False
            validation_results['validation_passed'] = False
        
        # Check for sex column (needed for waist circumference and HDL thresholds)
        if 'sex' not in df.columns:
            validation_results['data_quality_issues'].append("Sex column missing - using default thresholds")
        
        # Check for missing values in critical columns
        for col in required_columns:
            if col in df.columns:
                missing_pct = (df[col].isnull().sum() / len(df)) * 100
                if missing_pct > 50:
                    validation_results['data_quality_issues'].append(
                        f"High missing values in {col}: {missing_pct:.1f}%"
                    )
        
        logger.info(f"Metabolic syndrome data validation: {'PASSED' if validation_results['validation_passed'] else 'FAILED'}")
        
        return validation_results
