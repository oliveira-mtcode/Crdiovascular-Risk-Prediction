"""
Risk prediction model interface for cardiovascular risk prediction pipeline.

This module provides the main interface for cardiovascular risk prediction models,
including black box algorithm integration and batch scoring capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import logging
from abc import ABC, abstractmethod
from pathlib import Path
import joblib
import yaml
from datetime import datetime

logger = logging.getLogger(__name__)


class RiskPredictionModel(ABC):
    """
    Abstract base class for cardiovascular risk prediction models.
    
    Provides a standardized interface for risk prediction algorithms
    and batch scoring capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize risk prediction model with configuration.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        self.config = config
        self.model_config = config.get('model', {})
        self.algorithm_config = self.model_config.get('algorithm', {})
        self.model_name = self.algorithm_config.get('name', 'unknown')
        self.model_parameters = self.algorithm_config.get('parameters', {})
        
        # Model state
        self.is_fitted = False
        self.training_data_info = {}
        self.feature_importance = {}
        self.model_metadata = {}
        
        logger.info(f"Risk prediction model initialized: {self.model_name}")
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RiskPredictionModel':
        """
        Fit the risk prediction model to training data.
        
        Args:
            X: Feature matrix
            y: Target variable (cardiovascular risk)
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict cardiovascular risk for new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of risk predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict cardiovascular risk probabilities for new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of risk probabilities
        """
        pass
    
    def batch_score(self, data: pd.DataFrame, 
                   patient_id_col: str = 'patient_id',
                   output_format: str = 'dataframe') -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Perform batch scoring on a dataset.
        
        Args:
            data: DataFrame containing patient data
            patient_id_col: Name of the patient ID column
            output_format: Output format ('dataframe', 'dict')
            
        Returns:
            Risk predictions in specified format
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before batch scoring")
        
        logger.info(f"Performing batch scoring on {len(data)} patients")
        
        # Prepare features
        feature_columns = self._get_feature_columns(data)
        X = data[feature_columns]
        
        # Make predictions
        risk_scores = self.predict_proba(X)
        risk_predictions = self.predict(X)
        
        # Create results
        results = {
            'patient_id': data[patient_id_col].values if patient_id_col in data.columns else range(len(data)),
            'risk_score': risk_scores,
            'risk_prediction': risk_predictions,
            'risk_category': self._categorize_risk(risk_scores)
        }
        
        if output_format == 'dataframe':
            return pd.DataFrame(results)
        else:
            return results
    
    def _get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """
        Get feature columns for prediction.
        
        Args:
            data: DataFrame containing the data
            
        Returns:
            List of feature column names
        """
        # This should be implemented by subclasses based on their specific requirements
        # For now, return numeric columns
        return data.select_dtypes(include=[np.number]).columns.tolist()
    
    def _categorize_risk(self, risk_scores: np.ndarray) -> np.ndarray:
        """
        Categorize risk scores into risk levels.
        
        Args:
            risk_scores: Array of risk scores
            
        Returns:
            Array of risk categories
        """
        categories = np.full(len(risk_scores), 'Unknown')
        categories[risk_scores < 0.1] = 'Low'
        categories[(risk_scores >= 0.1) & (risk_scores < 0.2)] = 'Moderate'
        categories[risk_scores >= 0.2] = 'High'
        return categories
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        return self.feature_importance
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_name': self.model_name,
            'model_parameters': self.model_parameters,
            'is_fitted': self.is_fitted,
            'training_data_info': self.training_data_info,
            'feature_importance': self.feature_importance,
            'model_metadata': self.model_metadata,
            'created_at': datetime.now().isoformat()
        }
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model data
        model_data = {
            'model_info': self.get_model_info(),
            'model_parameters': self.model_parameters,
            'feature_importance': self.feature_importance,
            'model_metadata': self.model_metadata
        }
        
        # Save using joblib
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Union[str, Path]) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load model data
        model_data = joblib.load(filepath)
        
        # Restore model state
        self.model_parameters = model_data['model_parameters']
        self.feature_importance = model_data['feature_importance']
        self.model_metadata = model_data['model_metadata']
        self.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")


class BlackBoxRiskModel(RiskPredictionModel):
    """
    Black box risk prediction model wrapper.
    
    Allows integration of external risk prediction algorithms
    as black box functions.
    """
    
    def __init__(self, config: Dict[str, Any], 
                 prediction_function: Optional[Callable] = None):
        """
        Initialize black box risk model.
        
        Args:
            config: Configuration dictionary
            prediction_function: External prediction function
        """
        super().__init__(config)
        self.prediction_function = prediction_function
        self.required_features = self.model_parameters.get('required_features', [])
        self.feature_mapping = self.model_parameters.get('feature_mapping', {})
        
        if self.prediction_function is None:
            logger.warning("No prediction function provided - using default")
            self.prediction_function = self._default_prediction_function
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BlackBoxRiskModel':
        """
        Fit the black box model (no-op for black box models).
        
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
        logger.info("Black box model fitted (no training required)")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict cardiovascular risk using black box function.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of risk predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare features
        X_prepared = self._prepare_features(X)
        
        # Make predictions
        predictions = self.prediction_function(X_prepared)
        
        # Convert to binary predictions if needed
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            predictions = (predictions[:, 1] > 0.5).astype(int)
        elif predictions.max() <= 1.0 and predictions.min() >= 0.0:
            predictions = (predictions > 0.5).astype(int)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict cardiovascular risk probabilities using black box function.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of risk probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare features
        X_prepared = self._prepare_features(X)
        
        # Make predictions
        predictions = self.prediction_function(X_prepared)
        
        # Ensure probabilities are in correct format
        if predictions.ndim == 1:
            # Single column of probabilities
            probabilities = np.column_stack([1 - predictions, predictions])
        else:
            # Multiple columns (already in probability format)
            probabilities = predictions
        
        return probabilities[:, 1]  # Return positive class probabilities
    
    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for black box prediction.
        
        Args:
            X: Input feature matrix
            
        Returns:
            Prepared feature matrix
        """
        X_prepared = X.copy()
        
        # Apply feature mapping if specified
        if self.feature_mapping:
            X_prepared = X_prepared.rename(columns=self.feature_mapping)
        
        # Check for required features
        missing_features = [f for f in self.required_features if f not in X_prepared.columns]
        if missing_features:
            logger.warning(f"Missing required features: {missing_features}")
            # Fill missing features with default values
            for feature in missing_features:
                X_prepared[feature] = 0.0
        
        # Select only required features
        if self.required_features:
            X_prepared = X_prepared[self.required_features]
        
        return X_prepared
    
    def _default_prediction_function(self, X: pd.DataFrame) -> np.ndarray:
        """
        Default prediction function (returns random probabilities).
        
        Args:
            X: Feature matrix
            
        Returns:
            Random risk probabilities
        """
        logger.warning("Using default prediction function - returning random probabilities")
        return np.random.random(len(X))
    
    def _get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """
        Get feature columns for black box model.
        
        Args:
            data: DataFrame containing the data
            
        Returns:
            List of feature column names
        """
        if self.required_features:
            return [col for col in self.required_features if col in data.columns]
        else:
            return data.select_dtypes(include=[np.number]).columns.tolist()


class ModelEnsemble(RiskPredictionModel):
    """
    Ensemble of multiple risk prediction models.
    
    Combines predictions from multiple models using various ensemble methods.
    """
    
    def __init__(self, config: Dict[str, Any], models: List[RiskPredictionModel]):
        """
        Initialize model ensemble.
        
        Args:
            config: Configuration dictionary
            models: List of risk prediction models
        """
        super().__init__(config)
        self.models = models
        self.ensemble_method = self.model_parameters.get('ensemble_method', 'average')
        self.model_weights = self.model_parameters.get('model_weights', None)
        
        if self.model_weights is None:
            self.model_weights = [1.0 / len(models)] * len(models)
        
        logger.info(f"Model ensemble initialized with {len(models)} models")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ModelEnsemble':
        """
        Fit all models in the ensemble.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting ensemble models")
        
        for i, model in enumerate(self.models):
            logger.info(f"Fitting model {i+1}/{len(self.models)}")
            model.fit(X, y)
        
        # Store training data info
        self.training_data_info = {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'feature_names': X.columns.tolist(),
            'target_distribution': y.value_counts().to_dict(),
            'n_models': len(self.models)
        }
        
        self.is_fitted = True
        logger.info("Ensemble models fitted successfully")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using ensemble of models.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get predictions from all models
        model_predictions = []
        for model in self.models:
            pred = model.predict(X)
            model_predictions.append(pred)
        
        # Combine predictions
        if self.ensemble_method == 'average':
            ensemble_pred = np.average(model_predictions, axis=0, weights=self.model_weights)
        elif self.ensemble_method == 'majority_vote':
            ensemble_pred = np.round(np.average(model_predictions, axis=0, weights=self.model_weights))
        else:
            raise ValueError(f"Unsupported ensemble method: {self.ensemble_method}")
        
        return ensemble_pred.astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities using ensemble of models.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of ensemble probabilities
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get probabilities from all models
        model_probabilities = []
        for model in self.models:
            proba = model.predict_proba(X)
            model_probabilities.append(proba)
        
        # Combine probabilities
        ensemble_proba = np.average(model_probabilities, axis=0, weights=self.model_weights)
        
        return ensemble_proba
    
    def _get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """
        Get feature columns for ensemble model.
        
        Args:
            data: DataFrame containing the data
            
        Returns:
            List of feature column names
        """
        # Use features from the first model
        if self.models:
            return self.models[0]._get_feature_columns(data)
        else:
            return data.select_dtypes(include=[np.number]).columns.tolist()
