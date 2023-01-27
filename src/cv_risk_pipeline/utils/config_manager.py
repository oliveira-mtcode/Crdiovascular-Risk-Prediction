"""
Configuration management module for cardiovascular risk prediction pipeline.

This module provides comprehensive configuration management capabilities
including loading, validation, and dynamic configuration updates.
"""

import yaml
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from datetime import datetime
import copy

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages configuration for the cardiovascular risk prediction pipeline.
    
    Provides comprehensive configuration management including loading,
    validation, merging, and dynamic updates of configuration files.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to the main configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self.config = {}
        self.config_schema = self._get_config_schema()
        
        if self.config_path and self.config_path.exists():
            self.load_config(self.config_path)
        
        logger.info("Configuration manager initialized")
    
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Loaded configuration dictionary
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        logger.info(f"Loading configuration from {config_path}")
        
        try:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
            
            # Validate configuration
            self.validate_config()
            
            logger.info("Configuration loaded successfully")
            return self.config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def save_config(self, config_path: Union[str, Path], 
                   format: str = 'yaml') -> None:
        """
        Save configuration to file.
        
        Args:
            config_path: Path to save the configuration
            format: Output format ('yaml' or 'json')
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving configuration to {config_path}")
        
        try:
            if format.lower() == 'yaml':
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
            elif format.lower() == 'json':
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported output format: {format}")
            
            logger.info("Configuration saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def merge_config(self, other_config: Dict[str, Any], 
                    overwrite: bool = True) -> None:
        """
        Merge another configuration into the current configuration.
        
        Args:
            other_config: Configuration dictionary to merge
            overwrite: Whether to overwrite existing values
        """
        logger.info("Merging configuration")
        
        if overwrite:
            self.config = self._deep_merge(self.config, other_config)
        else:
            self.config = self._deep_merge(other_config, self.config)
        
        # Validate merged configuration
        self.validate_config()
        
        logger.info("Configuration merged successfully")
    
    def update_config(self, key_path: str, value: Any) -> None:
        """
        Update a specific configuration value.
        
        Args:
            key_path: Dot-separated path to the configuration key
            value: New value to set
        """
        logger.info(f"Updating configuration: {key_path} = {value}")
        
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value
        
        # Validate updated configuration
        self.validate_config()
        
        logger.info("Configuration updated successfully")
    
    def get_config(self, key_path: Optional[str] = None, 
                  default: Any = None) -> Any:
        """
        Get configuration value(s).
        
        Args:
            key_path: Dot-separated path to the configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or entire config if key_path is None
        """
        if key_path is None:
            return self.config
        
        keys = key_path.split('.')
        config = self.config
        
        try:
            for key in keys:
                config = config[key]
            return config
        except (KeyError, TypeError):
            return default
    
    def validate_config(self) -> bool:
        """
        Validate configuration against schema.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        logger.info("Validating configuration")
        
        try:
            self._validate_schema(self.config, self.config_schema)
            logger.info("Configuration validation passed")
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def create_default_config(self) -> Dict[str, Any]:
        """
        Create default configuration.
        
        Returns:
            Default configuration dictionary
        """
        logger.info("Creating default configuration")
        
        default_config = {
            'data': {
                'raw_data_path': 'data/raw',
                'processed_data_path': 'data/processed',
                'external_data_path': 'data/external',
                'output_path': 'outputs'
            },
            'metabolic_syndrome': {
                'criteria': {
                    'waist_circumference': {
                        'caucasian_male': 102,
                        'caucasian_female': 88,
                        'asian_male': 90,
                        'asian_female': 80
                    },
                    'blood_pressure': {
                        'systolic': 130,
                        'diastolic': 85
                    },
                    'fasting_glucose': 100,
                    'triglycerides': 150,
                    'hdl_cholesterol': {
                        'male': 40,
                        'female': 50
                    }
                },
                'required_criteria': 3
            },
            'data_cleaning': {
                'outlier_detection': {
                    'method': 'iqr',
                    'iqr_multiplier': 1.5,
                    'zscore_threshold': 3.0
                },
                'missing_values': {
                    'strategy': 'interpolate',
                    'max_missing_percentage': 0.3
                },
                'temporal': {
                    'min_observations_per_patient': 2,
                    'max_gap_days': 365
                }
            },
            'feature_engineering': {
                'temporal_features': {
                    'aggregations': ['mean', 'std', 'min', 'max', 'trend', 'variability'],
                    'time_windows': [30, 90, 180, 365]
                },
                'clinical_features': {
                    'risk_factors': [
                        'age', 'sex', 'smoking_status', 'diabetes_status',
                        'family_history_cvd', 'bmi', 'waist_circumference'
                    ],
                    'lab_values': [
                        'total_cholesterol', 'ldl_cholesterol', 'hdl_cholesterol',
                        'triglycerides', 'glucose', 'hba1c'
                    ],
                    'vital_signs': [
                        'systolic_bp', 'diastolic_bp', 'heart_rate'
                    ]
                }
            },
            'model': {
                'algorithm': {
                    'name': 'framingham_risk_score',
                    'parameters': {}
                },
                'cross_validation': {
                    'n_folds': 5,
                    'random_state': 42,
                    'stratify': True
                },
                'train_test_split': {
                    'test_size': 0.2,
                    'random_state': 42,
                    'stratify': True
                }
            },
            'validation': {
                'primary_metrics': [
                    'auc_roc', 'sensitivity', 'specificity', 'ppv', 'npv'
                ],
                'secondary_metrics': [
                    'accuracy', 'f1_score', 'precision', 'recall',
                    'calibration_slope', 'calibration_intercept'
                ],
                'calibration': {
                    'n_bins': 10,
                    'method': 'quantile'
                },
                'subgroup_analysis': {
                    'groups': ['age_group', 'sex', 'baseline_risk', 'metabolic_syndrome_severity'],
                    'age_groups': [[18, 40], [40, 60], [60, 80], [80, 100]],
                    'baseline_risk_groups': {
                        'low': [0, 0.1],
                        'moderate': [0.1, 0.2],
                        'high': [0.2, 1.0]
                    }
                }
            },
            'visualization': {
                'plots': {
                    'style': 'seaborn-v0_8',
                    'figure_size': [12, 8],
                    'dpi': 300
                },
                'colors': {
                    'primary': '#2E86AB',
                    'secondary': '#A23B72',
                    'success': '#F18F01',
                    'warning': '#C73E1D'
                },
                'output_formats': ['png', 'pdf', 'svg']
            },
            'reporting': {
                'formats': ['html', 'pdf', 'docx'],
                'templates': {
                    'html_template': 'templates/report_template.html',
                    'pdf_template': 'templates/report_template.tex'
                },
                'fda_510k': {
                    'include_algorithm_description': True,
                    'include_validation_protocol': True,
                    'include_statistical_analysis_plan': True,
                    'include_risk_benefit_analysis': True
                }
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'logs/pipeline.log',
                'max_file_size': '10MB',
                'backup_count': 5
            }
        }
        
        self.config = default_config
        return default_config
    
    def export_config(self, format: str = 'yaml') -> str:
        """
        Export configuration as string.
        
        Args:
            format: Output format ('yaml' or 'json')
            
        Returns:
            Configuration as string
        """
        if format.lower() == 'yaml':
            return yaml.dump(self.config, default_flow_style=False, indent=2)
        elif format.lower() == 'json':
            return json.dumps(self.config, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def import_config(self, config_string: str, format: str = 'yaml') -> None:
        """
        Import configuration from string.
        
        Args:
            config_string: Configuration as string
            format: Input format ('yaml' or 'json')
        """
        logger.info("Importing configuration from string")
        
        try:
            if format.lower() == 'yaml':
                self.config = yaml.safe_load(config_string)
            elif format.lower() == 'json':
                self.config = json.loads(config_string)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Validate imported configuration
            self.validate_config()
            
            logger.info("Configuration imported successfully")
            
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            raise
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get configuration summary.
        
        Returns:
            Configuration summary dictionary
        """
        return {
            'config_path': str(self.config_path) if self.config_path else None,
            'config_size': len(str(self.config)),
            'sections': list(self.config.keys()),
            'last_updated': datetime.now().isoformat(),
            'validation_status': 'valid' if self.validate_config() else 'invalid'
        }
    
    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            dict1: First dictionary
            dict2: Second dictionary
            
        Returns:
            Merged dictionary
        """
        result = copy.deepcopy(dict1)
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def _validate_schema(self, config: Dict[str, Any], schema: Dict[str, Any]) -> None:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration to validate
            schema: Schema to validate against
        """
        for key, expected_type in schema.items():
            if key not in config:
                raise KeyError(f"Missing required configuration key: {key}")
            
            if not isinstance(config[key], expected_type):
                raise TypeError(f"Configuration key '{key}' should be {expected_type}, got {type(config[key])}")
    
    def _get_config_schema(self) -> Dict[str, type]:
        """
        Get configuration schema.
        
        Returns:
            Configuration schema dictionary
        """
        return {
            'data': dict,
            'metabolic_syndrome': dict,
            'data_cleaning': dict,
            'feature_engineering': dict,
            'model': dict,
            'validation': dict,
            'visualization': dict,
            'reporting': dict,
            'logging': dict
        }
    
    def create_config_template(self, output_path: Union[str, Path]) -> None:
        """
        Create configuration template file.
        
        Args:
            output_path: Path to save the template
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create template with comments
        template = """# Cardiovascular Risk Prediction Pipeline Configuration
# This file contains all configuration parameters for the pipeline

# Data paths and directories
data:
  raw_data_path: "data/raw"                    # Path to raw data files
  processed_data_path: "data/processed"        # Path to processed data files
  external_data_path: "data/external"          # Path to external data files
  output_path: "outputs"                       # Path for output files

# Metabolic syndrome identification criteria (ATP III guidelines)
metabolic_syndrome:
  criteria:
    # Waist circumference thresholds (cm) - varies by ethnicity
    waist_circumference:
      caucasian_male: 102
      caucasian_female: 88
      asian_male: 90
      asian_female: 80
    
    # Blood pressure thresholds (mmHg)
    blood_pressure:
      systolic: 130
      diastolic: 85
    
    # Laboratory thresholds
    fasting_glucose: 100        # mg/dL
    triglycerides: 150          # mg/dL
    hdl_cholesterol:
      male: 40                  # mg/dL
      female: 50                # mg/dL
  
  required_criteria: 3          # Number of criteria required for diagnosis

# Data cleaning parameters
data_cleaning:
  outlier_detection:
    method: "iqr"               # Options: "iqr", "zscore", "isolation_forest"
    iqr_multiplier: 1.5
    zscore_threshold: 3.0
  
  missing_values:
    strategy: "interpolate"      # Options: "drop", "interpolate", "forward_fill", "backward_fill"
    max_missing_percentage: 0.3
  
  temporal:
    min_observations_per_patient: 2
    max_gap_days: 365

# Feature engineering parameters
feature_engineering:
  temporal_features:
    aggregations: ["mean", "std", "min", "max", "trend", "variability"]
    time_windows: [30, 90, 180, 365]  # days
  
  clinical_features:
    risk_factors:
      - "age"
      - "sex"
      - "smoking_status"
      - "diabetes_status"
      - "family_history_cvd"
      - "bmi"
      - "waist_circumference"
    
    lab_values:
      - "total_cholesterol"
      - "ldl_cholesterol"
      - "hdl_cholesterol"
      - "triglycerides"
      - "glucose"
      - "hba1c"
    
    vital_signs:
      - "systolic_bp"
      - "diastolic_bp"
      - "heart_rate"

# Model configuration
model:
  algorithm:
    name: "framingham_risk_score"  # Options: "framingham_risk_score", "pooled_cohort_equation", "custom"
    parameters: {}
  
  cross_validation:
    n_folds: 5
    random_state: 42
    stratify: true
  
  train_test_split:
    test_size: 0.2
    random_state: 42
    stratify: true

# Validation parameters
validation:
  primary_metrics:
    - "auc_roc"
    - "sensitivity"
    - "specificity"
    - "ppv"
    - "npv"
  
  secondary_metrics:
    - "accuracy"
    - "f1_score"
    - "precision"
    - "recall"
    - "calibration_slope"
    - "calibration_intercept"
  
  calibration:
    n_bins: 10
    method: "quantile"          # Options: "uniform", "quantile"
  
  subgroup_analysis:
    groups:
      - "age_group"
      - "sex"
      - "baseline_risk"
      - "metabolic_syndrome_severity"
    
    age_groups:
      - [18, 40]
      - [40, 60]
      - [60, 80]
      - [80, 100]
    
    baseline_risk_groups:
      low: [0, 0.1]
      moderate: [0.1, 0.2]
      high: [0.2, 1.0]

# Visualization settings
visualization:
  plots:
    style: "seaborn-v0_8"
    figure_size: [12, 8]
    dpi: 300
  
  colors:
    primary: "#2E86AB"
    secondary: "#A23B72"
    success: "#F18F01"
    warning: "#C73E1D"
  
  output_formats: ["png", "pdf", "svg"]

# Reporting settings
reporting:
  formats: ["html", "pdf", "docx"]
  
  templates:
    html_template: "templates/report_template.html"
    pdf_template: "templates/report_template.tex"
  
  fda_510k:
    include_algorithm_description: true
    include_validation_protocol: true
    include_statistical_analysis_plan: true
    include_risk_benefit_analysis: true

# Logging configuration
logging:
  level: "INFO"                 # Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/pipeline.log"
  max_file_size: "10MB"
  backup_count: 5
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template)
        
        logger.info(f"Configuration template created at {output_path}")
