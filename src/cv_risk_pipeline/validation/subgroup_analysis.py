"""
Subgroup analysis module for cardiovascular risk prediction pipeline.

This module provides comprehensive subgroup analysis capabilities for
validating model performance across different patient populations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from scipy import stats
import warnings

logger = logging.getLogger(__name__)


class SubgroupAnalysis:
    """
    Provides subgroup analysis capabilities for cardiovascular risk prediction models.
    
    Analyzes model performance across different patient subgroups including
    age groups, sex, baseline risk, and metabolic syndrome severity.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize subgroup analysis with configuration.
        
        Args:
            config: Configuration dictionary containing subgroup analysis parameters
        """
        self.config = config
        self.subgroup_config = config.get('validation', {}).get('subgroup_analysis', {})
        self.subgroups = self.subgroup_config.get('groups', [])
        self.age_groups = self.subgroup_config.get('age_groups', [])
        self.baseline_risk_groups = self.subgroup_config.get('baseline_risk_groups', {})
        
        logger.info("Subgroup analysis initialized")
    
    def analyze_subgroups(self, y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray,
                         data: pd.DataFrame, patient_id_col: str = 'patient_id') -> Dict[str, Any]:
        """
        Perform comprehensive subgroup analysis.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_scores: Prediction scores/probabilities
            data: DataFrame containing patient data
            patient_id_col: Name of the patient ID column
            
        Returns:
            Dictionary containing subgroup analysis results
        """
        logger.info("Performing subgroup analysis")
        
        # Create results dictionary
        results = {
            'subgroup_results': {},
            'statistical_tests': {},
            'summary_statistics': {}
        }
        
        # Analyze each subgroup
        for subgroup in self.subgroups:
            if subgroup in data.columns:
                logger.info(f"Analyzing subgroup: {subgroup}")
                subgroup_results = self._analyze_single_subgroup(
                    y_true, y_pred, y_scores, data, subgroup
                )
                results['subgroup_results'][subgroup] = subgroup_results
        
        # Perform statistical tests
        results['statistical_tests'] = self._perform_statistical_tests(
            y_true, y_pred, y_scores, data
        )
        
        # Generate summary statistics
        results['summary_statistics'] = self._generate_summary_statistics(
            y_true, y_pred, y_scores, data
        )
        
        logger.info("Subgroup analysis completed")
        return results
    
    def _analyze_single_subgroup(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               y_scores: np.ndarray, data: pd.DataFrame, 
                               subgroup_col: str) -> Dict[str, Any]:
        """
        Analyze performance for a single subgroup.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_scores: Prediction scores/probabilities
            data: DataFrame containing patient data
            subgroup_col: Name of the subgroup column
            
        Returns:
            Dictionary containing subgroup analysis results
        """
        subgroup_results = {}
        
        # Get unique values in subgroup
        unique_values = data[subgroup_col].unique()
        
        for value in unique_values:
            # Create mask for this subgroup
            mask = data[subgroup_col] == value
            
            if mask.sum() < 10:  # Skip subgroups with too few samples
                logger.warning(f"Skipping subgroup {subgroup_col}={value} (n={mask.sum()})")
                continue
            
            # Extract data for this subgroup
            y_true_sub = y_true[mask]
            y_pred_sub = y_pred[mask]
            y_scores_sub = y_scores[mask]
            
            # Compute performance metrics
            metrics = self._compute_subgroup_metrics(y_true_sub, y_pred_sub, y_scores_sub)
            
            # Add sample size information
            metrics['sample_size'] = len(y_true_sub)
            metrics['positive_samples'] = int(y_true_sub.sum())
            metrics['negative_samples'] = int(len(y_true_sub) - y_true_sub.sum())
            metrics['prevalence'] = float(y_true_sub.mean())
            
            subgroup_results[str(value)] = metrics
        
        return subgroup_results
    
    def _compute_subgroup_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                y_scores: np.ndarray) -> Dict[str, float]:
        """
        Compute performance metrics for a subgroup.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_scores: Prediction scores/probabilities
            
        Returns:
            Dictionary containing performance metrics
        """
        metrics = {}
        
        try:
            # AUC-ROC
            if len(np.unique(y_true)) > 1:  # Need both classes
                metrics['auc_roc'] = roc_auc_score(y_true, y_scores)
            else:
                metrics['auc_roc'] = np.nan
        except Exception as e:
            logger.warning(f"Could not compute AUC-ROC: {e}")
            metrics['auc_roc'] = np.nan
        
        try:
            # Accuracy
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
        except Exception as e:
            logger.warning(f"Could not compute accuracy: {e}")
            metrics['accuracy'] = np.nan
        
        try:
            # Precision
            metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        except Exception as e:
            logger.warning(f"Could not compute precision: {e}")
            metrics['precision'] = np.nan
        
        try:
            # Recall (Sensitivity)
            metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        except Exception as e:
            logger.warning(f"Could not compute recall: {e}")
            metrics['recall'] = np.nan
        
        try:
            # F1 Score
            metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        except Exception as e:
            logger.warning(f"Could not compute F1 score: {e}")
            metrics['f1_score'] = np.nan
        
        # Specificity
        try:
            from sklearn.metrics import confusion_matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        except Exception as e:
            logger.warning(f"Could not compute specificity: {e}")
            metrics['specificity'] = np.nan
        
        # Positive and Negative Predictive Values
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        except Exception as e:
            logger.warning(f"Could not compute PPV/NPV: {e}")
            metrics['ppv'] = np.nan
            metrics['npv'] = np.nan
        
        return metrics
    
    def _perform_statistical_tests(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_scores: np.ndarray, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform statistical tests for subgroup differences.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_scores: Prediction scores/probabilities
            data: DataFrame containing patient data
            
        Returns:
            Dictionary containing statistical test results
        """
        statistical_tests = {}
        
        # Test for differences across subgroups
        for subgroup in self.subgroups:
            if subgroup in data.columns:
                logger.info(f"Performing statistical tests for subgroup: {subgroup}")
                
                # Get unique values
                unique_values = data[subgroup].unique()
                
                if len(unique_values) < 2:
                    continue
                
                # Collect metrics for each subgroup
                subgroup_metrics = {}
                for value in unique_values:
                    mask = data[subgroup] == value
                    if mask.sum() < 10:  # Skip small subgroups
                        continue
                    
                    y_true_sub = y_true[mask]
                    y_pred_sub = y_pred[mask]
                    y_scores_sub = y_scores[mask]
                    
                    metrics = self._compute_subgroup_metrics(y_true_sub, y_pred_sub, y_scores_sub)
                    subgroup_metrics[str(value)] = metrics
                
                # Perform statistical tests
                test_results = self._test_subgroup_differences(subgroup_metrics, subgroup)
                statistical_tests[subgroup] = test_results
        
        return statistical_tests
    
    def _test_subgroup_differences(self, subgroup_metrics: Dict[str, Dict[str, float]], 
                                 subgroup_name: str) -> Dict[str, Any]:
        """
        Test for statistical differences between subgroups.
        
        Args:
            subgroup_metrics: Dictionary containing metrics for each subgroup
            subgroup_name: Name of the subgroup
            
        Returns:
            Dictionary containing statistical test results
        """
        test_results = {}
        
        # Extract metrics for testing
        metrics_to_test = ['auc_roc', 'accuracy', 'precision', 'recall', 'f1_score', 'specificity']
        
        for metric in metrics_to_test:
            # Collect metric values for each subgroup
            metric_values = []
            subgroup_names = []
            
            for subgroup, metrics in subgroup_metrics.items():
                if metric in metrics and not np.isnan(metrics[metric]):
                    metric_values.append(metrics[metric])
                    subgroup_names.append(subgroup)
            
            if len(metric_values) < 2:
                continue
            
            # Perform ANOVA test
            try:
                f_statistic, p_value = stats.f_oneway(*[metric_values])
                test_results[metric] = {
                    'f_statistic': float(f_statistic),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'subgroup_values': dict(zip(subgroup_names, metric_values))
                }
            except Exception as e:
                logger.warning(f"Could not perform ANOVA test for {metric}: {e}")
                test_results[metric] = {
                    'f_statistic': np.nan,
                    'p_value': np.nan,
                    'significant': False,
                    'subgroup_values': dict(zip(subgroup_names, metric_values))
                }
        
        return test_results
    
    def _generate_summary_statistics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_scores: np.ndarray, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for the overall dataset.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_scores: Prediction scores/probabilities
            data: DataFrame containing patient data
            
        Returns:
            Dictionary containing summary statistics
        """
        summary = {
            'overall_performance': self._compute_subgroup_metrics(y_true, y_pred, y_scores),
            'dataset_characteristics': {},
            'subgroup_distributions': {}
        }
        
        # Dataset characteristics
        summary['dataset_characteristics'] = {
            'total_samples': len(y_true),
            'positive_samples': int(y_true.sum()),
            'negative_samples': int(len(y_true) - y_true.sum()),
            'prevalence': float(y_true.mean()),
            'mean_predicted_risk': float(y_scores.mean()),
            'std_predicted_risk': float(y_scores.std())
        }
        
        # Subgroup distributions
        for subgroup in self.subgroups:
            if subgroup in data.columns:
                distribution = data[subgroup].value_counts().to_dict()
                summary['subgroup_distributions'][subgroup] = distribution
        
        return summary
    
    def analyze_age_groups(self, y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray,
                          data: pd.DataFrame, age_col: str = 'age') -> Dict[str, Any]:
        """
        Analyze performance across age groups.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_scores: Prediction scores/probabilities
            data: DataFrame containing patient data
            age_col: Name of the age column
            
        Returns:
            Dictionary containing age group analysis results
        """
        if age_col not in data.columns:
            logger.warning(f"Age column {age_col} not found in data")
            return {}
        
        logger.info("Analyzing age groups")
        
        # Create age groups
        data_with_age_groups = data.copy()
        data_with_age_groups['age_group'] = pd.cut(
            data[age_col], 
            bins=[0] + [group[1] for group in self.age_groups],
            labels=[f"{group[0]}-{group[1]}" for group in self.age_groups],
            include_lowest=True
        )
        
        # Analyze age groups
        age_group_results = self._analyze_single_subgroup(
            y_true, y_pred, y_scores, data_with_age_groups, 'age_group'
        )
        
        return age_group_results
    
    def analyze_baseline_risk_groups(self, y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray,
                                   data: pd.DataFrame, risk_col: str = 'baseline_risk') -> Dict[str, Any]:
        """
        Analyze performance across baseline risk groups.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_scores: Prediction scores/probabilities
            data: DataFrame containing patient data
            risk_col: Name of the baseline risk column
            
        Returns:
            Dictionary containing baseline risk group analysis results
        """
        if risk_col not in data.columns:
            logger.warning(f"Baseline risk column {risk_col} not found in data")
            return {}
        
        logger.info("Analyzing baseline risk groups")
        
        # Create baseline risk groups
        data_with_risk_groups = data.copy()
        risk_groups = []
        
        for risk_level, (min_risk, max_risk) in self.baseline_risk_groups.items():
            mask = (data[risk_col] >= min_risk) & (data[risk_col] < max_risk)
            data_with_risk_groups.loc[mask, 'baseline_risk_group'] = risk_level
        
        # Analyze baseline risk groups
        baseline_risk_results = self._analyze_single_subgroup(
            y_true, y_pred, y_scores, data_with_risk_groups, 'baseline_risk_group'
        )
        
        return baseline_risk_results
    
    def generate_subgroup_report(self, subgroup_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive subgroup analysis report.
        
        Args:
            subgroup_results: Dictionary containing subgroup analysis results
            
        Returns:
            Dictionary containing formatted subgroup report
        """
        logger.info("Generating subgroup analysis report")
        
        report = {
            'summary': {
                'analysis_date': pd.Timestamp.now().isoformat(),
                'subgroups_analyzed': list(subgroup_results.get('subgroup_results', {}).keys()),
                'statistical_tests_performed': list(subgroup_results.get('statistical_tests', {}).keys())
            },
            'subgroup_results': subgroup_results.get('subgroup_results', {}),
            'statistical_tests': subgroup_results.get('statistical_tests', {}),
            'summary_statistics': subgroup_results.get('summary_statistics', {}),
            'interpretations': self._interpret_subgroup_results(subgroup_results)
        }
        
        logger.info("Subgroup analysis report generated")
        return report
    
    def _interpret_subgroup_results(self, subgroup_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Interpret subgroup analysis results.
        
        Args:
            subgroup_results: Dictionary containing subgroup analysis results
            
        Returns:
            Dictionary containing interpretations
        """
        interpretations = {}
        
        # Analyze statistical significance
        statistical_tests = subgroup_results.get('statistical_tests', {})
        significant_differences = []
        
        for subgroup, tests in statistical_tests.items():
            for metric, test_result in tests.items():
                if test_result.get('significant', False):
                    significant_differences.append(f"{subgroup} - {metric}")
        
        if significant_differences:
            interpretations['statistical_significance'] = (
                f"Significant differences found in: {', '.join(significant_differences)}"
            )
        else:
            interpretations['statistical_significance'] = (
                "No significant differences found between subgroups"
            )
        
        # Analyze performance consistency
        subgroup_results_data = subgroup_results.get('subgroup_results', {})
        performance_consistency = []
        
        for subgroup, results in subgroup_results_data.items():
            auc_values = []
            for subgroup_value, metrics in results.items():
                if 'auc_roc' in metrics and not np.isnan(metrics['auc_roc']):
                    auc_values.append(metrics['auc_roc'])
            
            if len(auc_values) > 1:
                auc_std = np.std(auc_values)
                if auc_std < 0.05:
                    performance_consistency.append(f"{subgroup} (consistent)")
                else:
                    performance_consistency.append(f"{subgroup} (variable)")
        
        if performance_consistency:
            interpretations['performance_consistency'] = (
                f"Performance consistency: {', '.join(performance_consistency)}"
            )
        
        return interpretations
