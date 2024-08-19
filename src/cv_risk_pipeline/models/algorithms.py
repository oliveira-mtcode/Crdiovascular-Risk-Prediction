"""
Risk prediction algorithms for cardiovascular risk prediction pipeline.

This module provides implementations of established cardiovascular risk prediction
algorithms including Framingham Risk Score and Pooled Cohort Equation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from .risk_prediction import RiskPredictionModel
import warnings

logger = logging.getLogger(__name__)


class FraminghamRiskScore(RiskPredictionModel):
    """
    Framingham Risk Score implementation for cardiovascular risk prediction.
    
    Implements the traditional Framingham Risk Score algorithm for
    10-year cardiovascular risk prediction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Framingham Risk Score model.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.model_name = "Framingham Risk Score"
        
        # Framingham Risk Score parameters
        self.age_coefficients = {
            'male': {'age': 0.04826, 'age_squared': 0.0},
            'female': {'age': 0.33766, 'age_squared': 0.0}
        }
        
        self.risk_factors = {
            'male': {
                'total_cholesterol': 0.02344,
                'hdl_cholesterol': -0.65945,
                'systolic_bp': 0.01817,
                'smoking': 0.52873,
                'diabetes': 0.42839
            },
            'female': {
                'total_cholesterol': 0.01282,
                'hdl_cholesterol': -0.70833,
                'systolic_bp': 0.02266,
                'smoking': 0.69154,
                'diabetes': 0.81073
            }
        }
        
        self.baseline_survival = {
            'male': 0.88936,
            'female': 0.95012
        }
        
        logger.info("Framingham Risk Score model initialized")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FraminghamRiskScore':
        """
        Fit the Framingham Risk Score model (no training required).
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Self for method chaining
        """
        # Store training data info
        self.training_data_info = {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'feature_names': X.columns.tolist(),
            'target_distribution': y.value_counts().to_dict()
        }
        
        self.is_fitted = True
        logger.info("Framingham Risk Score model fitted")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict cardiovascular risk using Framingham Risk Score.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of risk predictions (binary)
        """
        probabilities = self.predict_proba(X)
        return (probabilities > 0.1).astype(int)  # 10% risk threshold
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict cardiovascular risk probabilities using Framingham Risk Score.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of risk probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Required features for Framingham Risk Score
        required_features = ['age', 'sex', 'total_cholesterol', 'hdl_cholesterol', 
                           'systolic_bp', 'smoking_status', 'diabetes_status']
        
        # Check for required features
        missing_features = [f for f in required_features if f not in X.columns]
        if missing_features:
            logger.warning(f"Missing features for Framingham Risk Score: {missing_features}")
            # Fill missing features with default values
            for feature in missing_features:
                if feature not in X.columns:
                    X = X.copy()
                    if feature == 'sex':
                        X[feature] = 'male'  # Default to male
                    elif feature in ['smoking_status', 'diabetes_status']:
                        X[feature] = 'no'  # Default to no
                    else:
                        X[feature] = 0.0  # Default to 0
        
        # Calculate risk scores
        risk_scores = []
        
        for idx, row in X.iterrows():
            try:
                # Get patient characteristics
                age = float(row['age'])
                sex = str(row['sex']).lower()
                total_chol = float(row['total_cholesterol'])
                hdl_chol = float(row['hdl_cholesterol'])
                systolic_bp = float(row['systolic_bp'])
                smoking = str(row['smoking_status']).lower() in ['yes', 'current', '1', 'true']
                diabetes = str(row['diabetes_status']).lower() in ['yes', '1', 'true']
                
                # Determine sex coefficient
                sex_key = 'male' if sex in ['male', 'm', '1'] else 'female'
                
                # Calculate linear predictor
                linear_predictor = 0.0
                
                # Age terms
                linear_predictor += self.age_coefficients[sex_key]['age'] * age
                if self.age_coefficients[sex_key]['age_squared'] != 0:
                    linear_predictor += self.age_coefficients[sex_key]['age_squared'] * (age ** 2)
                
                # Risk factor terms
                linear_predictor += self.risk_factors[sex_key]['total_cholesterol'] * total_chol
                linear_predictor += self.risk_factors[sex_key]['hdl_cholesterol'] * hdl_chol
                linear_predictor += self.risk_factors[sex_key]['systolic_bp'] * systolic_bp
                
                if smoking:
                    linear_predictor += self.risk_factors[sex_key]['smoking']
                
                if diabetes:
                    linear_predictor += self.risk_factors[sex_key]['diabetes']
                
                # Calculate 10-year risk
                risk = 1 - (self.baseline_survival[sex_key] ** np.exp(linear_predictor))
                risk = max(0.0, min(1.0, risk))  # Clamp to [0, 1]
                
                risk_scores.append(risk)
                
            except Exception as e:
                logger.warning(f"Error calculating Framingham Risk Score for patient {idx}: {e}")
                risk_scores.append(0.0)  # Default to 0 risk
        
        return np.array(risk_scores)
    
    def _get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """
        Get feature columns for Framingham Risk Score.
        
        Args:
            data: DataFrame containing the data
            
        Returns:
            List of required feature column names
        """
        required_features = ['age', 'sex', 'total_cholesterol', 'hdl_cholesterol', 
                           'systolic_bp', 'smoking_status', 'diabetes_status']
        return [col for col in required_features if col in data.columns]


class PooledCohortEquation(RiskPredictionModel):
    """
    Pooled Cohort Equation implementation for cardiovascular risk prediction.
    
    Implements the 2013 ACC/AHA Pooled Cohort Equation for
    10-year cardiovascular risk prediction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Pooled Cohort Equation model.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.model_name = "Pooled Cohort Equation"
        
        # Pooled Cohort Equation coefficients
        self.coefficients = {
            'white_male': {
                'age': 0.04826,
                'total_cholesterol': 0.02344,
                'hdl_cholesterol': -0.65945,
                'systolic_bp': 0.01817,
                'smoking': 0.52873,
                'diabetes': 0.42839
            },
            'white_female': {
                'age': 0.33766,
                'total_cholesterol': 0.01282,
                'hdl_cholesterol': -0.70833,
                'systolic_bp': 0.02266,
                'smoking': 0.69154,
                'diabetes': 0.81073
            },
            'black_male': {
                'age': 0.04826,
                'total_cholesterol': 0.02344,
                'hdl_cholesterol': -0.65945,
                'systolic_bp': 0.01817,
                'smoking': 0.52873,
                'diabetes': 0.42839
            },
            'black_female': {
                'age': 0.33766,
                'total_cholesterol': 0.01282,
                'hdl_cholesterol': -0.70833,
                'systolic_bp': 0.02266,
                'smoking': 0.69154,
                'diabetes': 0.81073
            }
        }
        
        self.baseline_survival = {
            'white_male': 0.88936,
            'white_female': 0.95012,
            'black_male': 0.88936,
            'black_female': 0.95012
        }
        
        logger.info("Pooled Cohort Equation model initialized")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'PooledCohortEquation':
        """
        Fit the Pooled Cohort Equation model (no training required).
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Self for method chaining
        """
        # Store training data info
        self.training_data_info = {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'feature_names': X.columns.tolist(),
            'target_distribution': y.value_counts().to_dict()
        }
        
        self.is_fitted = True
        logger.info("Pooled Cohort Equation model fitted")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict cardiovascular risk using Pooled Cohort Equation.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of risk predictions (binary)
        """
        probabilities = self.predict_proba(X)
        return (probabilities > 0.075).astype(int)  # 7.5% risk threshold
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict cardiovascular risk probabilities using Pooled Cohort Equation.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of risk probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Required features for Pooled Cohort Equation
        required_features = ['age', 'sex', 'race', 'total_cholesterol', 'hdl_cholesterol', 
                           'systolic_bp', 'smoking_status', 'diabetes_status']
        
        # Check for required features
        missing_features = [f for f in required_features if f not in X.columns]
        if missing_features:
            logger.warning(f"Missing features for Pooled Cohort Equation: {missing_features}")
            # Fill missing features with default values
            for feature in missing_features:
                if feature not in X.columns:
                    X = X.copy()
                    if feature == 'sex':
                        X[feature] = 'male'  # Default to male
                    elif feature == 'race':
                        X[feature] = 'white'  # Default to white
                    elif feature in ['smoking_status', 'diabetes_status']:
                        X[feature] = 'no'  # Default to no
                    else:
                        X[feature] = 0.0  # Default to 0
        
        # Calculate risk scores
        risk_scores = []
        
        for idx, row in X.iterrows():
            try:
                # Get patient characteristics
                age = float(row['age'])
                sex = str(row['sex']).lower()
                race = str(row['race']).lower()
                total_chol = float(row['total_cholesterol'])
                hdl_chol = float(row['hdl_cholesterol'])
                systolic_bp = float(row['systolic_bp'])
                smoking = str(row['smoking_status']).lower() in ['yes', 'current', '1', 'true']
                diabetes = str(row['diabetes_status']).lower() in ['yes', '1', 'true']
                
                # Determine race and sex
                race_key = 'black' if race in ['black', 'african american', 'african-american'] else 'white'
                sex_key = 'male' if sex in ['male', 'm', '1'] else 'female'
                
                # Get coefficient key
                coeff_key = f"{race_key}_{sex_key}"
                
                # Calculate linear predictor
                linear_predictor = 0.0
                
                # Age term
                linear_predictor += self.coefficients[coeff_key]['age'] * age
                
                # Risk factor terms
                linear_predictor += self.coefficients[coeff_key]['total_cholesterol'] * total_chol
                linear_predictor += self.coefficients[coeff_key]['hdl_cholesterol'] * hdl_chol
                linear_predictor += self.coefficients[coeff_key]['systolic_bp'] * systolic_bp
                
                if smoking:
                    linear_predictor += self.coefficients[coeff_key]['smoking']
                
                if diabetes:
                    linear_predictor += self.coefficients[coeff_key]['diabetes']
                
                # Calculate 10-year risk
                risk = 1 - (self.baseline_survival[coeff_key] ** np.exp(linear_predictor))
                risk = max(0.0, min(1.0, risk))  # Clamp to [0, 1]
                
                risk_scores.append(risk)
                
            except Exception as e:
                logger.warning(f"Error calculating Pooled Cohort Equation for patient {idx}: {e}")
                risk_scores.append(0.0)  # Default to 0 risk
        
        return np.array(risk_scores)
    
    def _get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """
        Get feature columns for Pooled Cohort Equation.
        
        Args:
            data: DataFrame containing the data
            
        Returns:
            List of required feature column names
        """
        required_features = ['age', 'sex', 'race', 'total_cholesterol', 'hdl_cholesterol', 
                           'systolic_bp', 'smoking_status', 'diabetes_status']
        return [col for col in required_features if col in data.columns]


class CustomRiskModel(RiskPredictionModel):
    """
    Custom risk prediction model for cardiovascular risk prediction.
    
    Provides a flexible framework for implementing custom risk prediction
    algorithms or integrating external models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize custom risk model.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.model_name = "Custom Risk Model"
        
        # Custom model parameters
        self.model_type = self.model_parameters.get('model_type', 'logistic_regression')
        self.feature_weights = self.model_parameters.get('feature_weights', {})
        self.risk_threshold = self.model_parameters.get('risk_threshold', 0.1)
        
        # Initialize model based on type
        self.model = None
        self._initialize_model()
        
        logger.info(f"Custom risk model initialized: {self.model_type}")
    
    def _initialize_model(self):
        """Initialize the underlying model based on configuration."""
        if self.model_type == 'logistic_regression':
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(random_state=42)
        elif self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(random_state=42)
        elif self.model_type == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(random_state=42)
        else:
            logger.warning(f"Unknown model type: {self.model_type}, using logistic regression")
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(random_state=42)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'CustomRiskModel':
        """
        Fit the custom risk model.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting custom risk model: {self.model_type}")
        
        # Fit the model
        self.model.fit(X, y)
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            self.feature_importance = dict(zip(X.columns, np.abs(self.model.coef_[0])))
        
        # Store training data info
        self.training_data_info = {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'feature_names': X.columns.tolist(),
            'target_distribution': y.value_counts().to_dict(),
            'model_type': self.model_type
        }
        
        self.is_fitted = True
        logger.info("Custom risk model fitted successfully")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict cardiovascular risk using custom model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of risk predictions (binary)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        probabilities = self.predict_proba(X)
        return (probabilities > self.risk_threshold).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict cardiovascular risk probabilities using custom model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of risk probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get probabilities from the model
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)[:, 1]
        else:
            # Fallback to decision function
            decision_scores = self.model.decision_function(X)
            probabilities = 1 / (1 + np.exp(-decision_scores))
        
        return probabilities
    
    def _get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """
        Get feature columns for custom model.
        
        Args:
            data: DataFrame containing the data
            
        Returns:
            List of feature column names
        """
        # Use all numeric columns by default
        return data.select_dtypes(include=[np.number]).columns.tolist()



