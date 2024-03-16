"""
Statistical validation module for cardiovascular risk prediction pipeline.

This module provides comprehensive statistical validation capabilities including
performance metrics, calibration analysis, and cross-validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    brier_score_loss, log_loss
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import calibration_curve
import warnings

logger = logging.getLogger(__name__)


class StatisticalValidation:
    """
    Provides comprehensive statistical validation for cardiovascular risk prediction models.
    
    Implements standard validation metrics, calibration analysis, and cross-validation
    for clinical decision support systems.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize statistical validation with configuration.
        
        Args:
            config: Configuration dictionary containing validation parameters
        """
        self.config = config
        self.validation_config = config.get('validation', {})
        self.primary_metrics = self.validation_config.get('primary_metrics', [])
        self.secondary_metrics = self.validation_config.get('secondary_metrics', [])
        self.calibration_config = self.validation_config.get('calibration', {})
        self.cv_config = config.get('model', {}).get('cross_validation', {})
        
        logger.info("Statistical validation initialized")
    
    def compute_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_scores: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive performance metrics.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_scores: Prediction scores/probabilities
            
        Returns:
            Dictionary containing performance metrics
        """
        logger.info("Computing performance metrics")
        
        metrics = {}
        
        # Primary metrics
        if 'auc_roc' in self.primary_metrics:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_scores)
            except ValueError as e:
                logger.warning(f"Could not compute AUC-ROC: {e}")
                metrics['auc_roc'] = np.nan
        
        if 'sensitivity' in self.primary_metrics:
            metrics['sensitivity'] = recall_score(y_true, y_pred, zero_division=0)
        
        if 'specificity' in self.primary_metrics:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        if 'ppv' in self.primary_metrics:
            metrics['ppv'] = precision_score(y_true, y_pred, zero_division=0)
        
        if 'npv' in self.primary_metrics:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Secondary metrics
        if 'accuracy' in self.secondary_metrics:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        if 'f1_score' in self.secondary_metrics:
            metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        if 'precision' in self.secondary_metrics:
            metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        
        if 'recall' in self.secondary_metrics:
            metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        
        # Calibration metrics
        if 'calibration_slope' in self.secondary_metrics:
            try:
                slope, intercept = self._compute_calibration_slope(y_true, y_scores)
                metrics['calibration_slope'] = slope
                metrics['calibration_intercept'] = intercept
            except Exception as e:
                logger.warning(f"Could not compute calibration slope: {e}")
                metrics['calibration_slope'] = np.nan
                metrics['calibration_intercept'] = np.nan
        
        # Additional metrics
        try:
            metrics['brier_score'] = brier_score_loss(y_true, y_scores)
        except Exception as e:
            logger.warning(f"Could not compute Brier score: {e}")
            metrics['brier_score'] = np.nan
        
        try:
            metrics['log_loss'] = log_loss(y_true, y_scores)
        except Exception as e:
            logger.warning(f"Could not compute log loss: {e}")
            metrics['log_loss'] = np.nan
        
        # Precision-Recall AUC
        try:
            metrics['auc_pr'] = average_precision_score(y_true, y_scores)
        except Exception as e:
            logger.warning(f"Could not compute PR-AUC: {e}")
            metrics['auc_pr'] = np.nan
        
        logger.info(f"Computed {len(metrics)} performance metrics")
        return metrics
    
    def _compute_calibration_slope(self, y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
        """
        Compute calibration slope and intercept.
        
        Args:
            y_true: True binary labels
            y_scores: Prediction scores/probabilities
            
        Returns:
            Tuple of (slope, intercept)
        """
        # Use calibration curve to get binned probabilities
        n_bins = self.calibration_config.get('n_bins', 10)
        method = self.calibration_config.get('method', 'quantile')
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_scores, n_bins=n_bins, strategy=method
        )
        
        # Fit linear regression
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()
        reg.fit(mean_predicted_value.reshape(-1, 1), fraction_of_positives)
        
        return reg.coef_[0], reg.intercept_
    
    def compute_confusion_matrix_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Compute detailed confusion matrix metrics.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            
        Returns:
            Dictionary containing confusion matrix and derived metrics
        """
        logger.info("Computing confusion matrix metrics")
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Extract values
        tn, fp, fn, tp = cm.ravel()
        
        # Compute metrics
        metrics = {
            'confusion_matrix': cm.tolist(),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'total_samples': int(tn + fp + fn + tp)
        }
        
        # Derived metrics
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['positive_predictive_value'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        
        # Likelihood ratios
        metrics['positive_likelihood_ratio'] = (
            metrics['sensitivity'] / (1 - metrics['specificity']) 
            if (1 - metrics['specificity']) > 0 else np.inf
        )
        metrics['negative_likelihood_ratio'] = (
            (1 - metrics['sensitivity']) / metrics['specificity'] 
            if metrics['specificity'] > 0 else np.inf
        )
        
        logger.info("Confusion matrix metrics computed")
        return metrics
    
    def compute_calibration_metrics(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, Any]:
        """
        Compute calibration metrics and statistics.
        
        Args:
            y_true: True binary labels
            y_scores: Prediction scores/probabilities
            
        Returns:
            Dictionary containing calibration metrics
        """
        logger.info("Computing calibration metrics")
        
        n_bins = self.calibration_config.get('n_bins', 10)
        method = self.calibration_config.get('method', 'quantile')
        
        # Compute calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_scores, n_bins=n_bins, strategy=method
        )
        
        # Compute calibration metrics
        metrics = {
            'calibration_curve': {
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist()
            },
            'n_bins': n_bins,
            'method': method
        }
        
        # Hosmer-Lemeshow test (simplified version)
        try:
            hl_statistic, hl_p_value = self._hosmer_lemeshow_test(y_true, y_scores, n_bins)
            metrics['hosmer_lemeshow_statistic'] = hl_statistic
            metrics['hosmer_lemeshow_p_value'] = hl_p_value
        except Exception as e:
            logger.warning(f"Could not compute Hosmer-Lemeshow test: {e}")
            metrics['hosmer_lemeshow_statistic'] = np.nan
            metrics['hosmer_lemeshow_p_value'] = np.nan
        
        # Calibration slope and intercept
        try:
            slope, intercept = self._compute_calibration_slope(y_true, y_scores)
            metrics['calibration_slope'] = slope
            metrics['calibration_intercept'] = intercept
        except Exception as e:
            logger.warning(f"Could not compute calibration slope: {e}")
            metrics['calibration_slope'] = np.nan
            metrics['calibration_intercept'] = np.nan
        
        # Brier score
        try:
            metrics['brier_score'] = brier_score_loss(y_true, y_scores)
        except Exception as e:
            logger.warning(f"Could not compute Brier score: {e}")
            metrics['brier_score'] = np.nan
        
        logger.info("Calibration metrics computed")
        return metrics
    
    def _hosmer_lemeshow_test(self, y_true: np.ndarray, y_scores: np.ndarray, n_bins: int) -> Tuple[float, float]:
        """
        Compute Hosmer-Lemeshow goodness-of-fit test.
        
        Args:
            y_true: True binary labels
            y_scores: Prediction scores/probabilities
            n_bins: Number of bins
            
        Returns:
            Tuple of (statistic, p_value)
        """
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        # Calculate observed and expected frequencies
        observed = []
        expected = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_scores > bin_lower) & (y_scores <= bin_upper)
            if in_bin.sum() > 0:
                observed_positive = y_true[in_bin].sum()
                expected_positive = y_scores[in_bin].sum()
                observed.append(observed_positive)
                expected.append(expected_positive)
        
        # Calculate chi-square statistic
        observed = np.array(observed)
        expected = np.array(expected)
        
        # Avoid division by zero
        expected = np.where(expected == 0, 1e-10, expected)
        
        chi_square = np.sum((observed - expected) ** 2 / expected)
        
        # Degrees of freedom (number of bins - 2)
        df = len(observed) - 2
        
        # P-value (simplified - in practice, use scipy.stats.chi2)
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(chi_square, df)
        
        return chi_square, p_value
    
    def perform_cross_validation(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Perform cross-validation analysis.
        
        Args:
            model: Trained model object
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary containing cross-validation results
        """
        logger.info("Performing cross-validation analysis")
        
        n_folds = self.cv_config.get('n_folds', 5)
        random_state = self.cv_config.get('random_state', 42)
        stratify = self.cv_config.get('stratify', True)
        
        # Create cross-validation strategy
        if stratify:
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        # Perform cross-validation
        cv_results = {}
        
        # AUC-ROC
        try:
            auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
            cv_results['auc_roc'] = {
                'scores': auc_scores.tolist(),
                'mean': auc_scores.mean(),
                'std': auc_scores.std(),
                'ci_95': (auc_scores.mean() - 1.96 * auc_scores.std(), 
                         auc_scores.mean() + 1.96 * auc_scores.std())
            }
        except Exception as e:
            logger.warning(f"Could not compute CV AUC-ROC: {e}")
            cv_results['auc_roc'] = {'scores': [], 'mean': np.nan, 'std': np.nan, 'ci_95': (np.nan, np.nan)}
        
        # Accuracy
        try:
            accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            cv_results['accuracy'] = {
                'scores': accuracy_scores.tolist(),
                'mean': accuracy_scores.mean(),
                'std': accuracy_scores.std(),
                'ci_95': (accuracy_scores.mean() - 1.96 * accuracy_scores.std(), 
                         accuracy_scores.mean() + 1.96 * accuracy_scores.std())
            }
        except Exception as e:
            logger.warning(f"Could not compute CV accuracy: {e}")
            cv_results['accuracy'] = {'scores': [], 'mean': np.nan, 'std': np.nan, 'ci_95': (np.nan, np.nan)}
        
        # F1 score
        try:
            f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
            cv_results['f1_score'] = {
                'scores': f1_scores.tolist(),
                'mean': f1_scores.mean(),
                'std': f1_scores.std(),
                'ci_95': (f1_scores.mean() - 1.96 * f1_scores.std(), 
                         f1_scores.mean() + 1.96 * f1_scores.std())
            }
        except Exception as e:
            logger.warning(f"Could not compute CV F1 score: {e}")
            cv_results['f1_score'] = {'scores': [], 'mean': np.nan, 'std': np.nan, 'ci_95': (np.nan, np.nan)}
        
        # Precision
        try:
            precision_scores = cross_val_score(model, X, y, cv=cv, scoring='precision')
            cv_results['precision'] = {
                'scores': precision_scores.tolist(),
                'mean': precision_scores.mean(),
                'std': precision_scores.std(),
                'ci_95': (precision_scores.mean() - 1.96 * precision_scores.std(), 
                         precision_scores.mean() + 1.96 * precision_scores.std())
            }
        except Exception as e:
            logger.warning(f"Could not compute CV precision: {e}")
            cv_results['precision'] = {'scores': [], 'mean': np.nan, 'std': np.nan, 'ci_95': (np.nan, np.nan)}
        
        # Recall
        try:
            recall_scores = cross_val_score(model, X, y, cv=cv, scoring='recall')
            cv_results['recall'] = {
                'scores': recall_scores.tolist(),
                'mean': recall_scores.mean(),
                'std': recall_scores.std(),
                'ci_95': (recall_scores.mean() - 1.96 * recall_scores.std(), 
                         recall_scores.mean() + 1.96 * recall_scores.std())
            }
        except Exception as e:
            logger.warning(f"Could not compute CV recall: {e}")
            cv_results['recall'] = {'scores': [], 'mean': np.nan, 'std': np.nan, 'ci_95': (np.nan, np.nan)}
        
        logger.info(f"Cross-validation completed with {n_folds} folds")
        return cv_results
    
    def generate_validation_report(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Args:
            validation_results: Dictionary containing validation results
            
        Returns:
            Dictionary containing formatted validation report
        """
        logger.info("Generating validation report")
        
        report = {
            'summary': {
                'validation_date': pd.Timestamp.now().isoformat(),
                'total_samples': validation_results.get('total_samples', 0),
                'positive_samples': validation_results.get('positive_samples', 0),
                'negative_samples': validation_results.get('negative_samples', 0),
                'prevalence': validation_results.get('prevalence', 0)
            },
            'performance_metrics': validation_results.get('performance_metrics', {}),
            'confusion_matrix': validation_results.get('confusion_matrix', {}),
            'calibration_metrics': validation_results.get('calibration_metrics', {}),
            'cross_validation': validation_results.get('cross_validation', {}),
            'interpretation': self._interpret_results(validation_results)
        }
        
        logger.info("Validation report generated")
        return report
    
    def _interpret_results(self, validation_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Interpret validation results and provide clinical context.
        
        Args:
            validation_results: Dictionary containing validation results
            
        Returns:
            Dictionary containing interpretations
        """
        interpretations = {}
        
        # AUC-ROC interpretation
        auc_roc = validation_results.get('performance_metrics', {}).get('auc_roc', np.nan)
        if not np.isnan(auc_roc):
            if auc_roc >= 0.9:
                interpretations['auc_roc'] = "Excellent discriminative ability"
            elif auc_roc >= 0.8:
                interpretations['auc_roc'] = "Good discriminative ability"
            elif auc_roc >= 0.7:
                interpretations['auc_roc'] = "Fair discriminative ability"
            else:
                interpretations['auc_roc'] = "Poor discriminative ability"
        
        # Calibration interpretation
        calibration_slope = validation_results.get('calibration_metrics', {}).get('calibration_slope', np.nan)
        if not np.isnan(calibration_slope):
            if 0.8 <= calibration_slope <= 1.2:
                interpretations['calibration'] = "Well-calibrated model"
            elif calibration_slope < 0.8:
                interpretations['calibration'] = "Overconfident model (overestimates risk)"
            else:
                interpretations['calibration'] = "Underconfident model (underestimates risk)"
        
        # Sensitivity and specificity interpretation
        sensitivity = validation_results.get('performance_metrics', {}).get('sensitivity', np.nan)
        specificity = validation_results.get('performance_metrics', {}).get('specificity', np.nan)
        
        if not np.isnan(sensitivity) and not np.isnan(specificity):
            if sensitivity >= 0.8 and specificity >= 0.8:
                interpretations['diagnostic_performance'] = "High sensitivity and specificity"
            elif sensitivity >= 0.8:
                interpretations['diagnostic_performance'] = "High sensitivity, moderate specificity"
            elif specificity >= 0.8:
                interpretations['diagnostic_performance'] = "Moderate sensitivity, high specificity"
            else:
                interpretations['diagnostic_performance'] = "Moderate sensitivity and specificity"
        
        return interpretations


