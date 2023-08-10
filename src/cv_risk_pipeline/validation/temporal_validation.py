"""
Temporal validation module for cardiovascular risk prediction pipeline.

This module provides temporal validation capabilities including
performance assessment over time and re-identification bias analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from scipy import stats
import warnings

logger = logging.getLogger(__name__)


class TemporalValidation:
    """
    Provides temporal validation capabilities for cardiovascular risk prediction models.
    
    Analyzes model performance over time and performs re-identification bias analysis
    for longitudinal clinical data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize temporal validation with configuration.
        
        Args:
            config: Configuration dictionary containing temporal validation parameters
        """
        self.config = config
        self.temporal_config = config.get('temporal_validation', {})
        self.time_windows = self.temporal_config.get('time_windows', [30, 90, 180, 365])
        self.min_samples_per_window = self.temporal_config.get('min_samples_per_window', 50)
        
        logger.info("Temporal validation initialized")
    
    def analyze_temporal_performance(self, y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray,
                                   data: pd.DataFrame, date_col: str = 'date',
                                   patient_id_col: str = 'patient_id') -> Dict[str, Any]:
        """
        Analyze model performance over time.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_scores: Prediction scores/probabilities
            data: DataFrame containing temporal data
            date_col: Name of the date column
            patient_id_col: Name of the patient ID column
            
        Returns:
            Dictionary containing temporal performance analysis results
        """
        logger.info("Analyzing temporal performance")
        
        # Ensure date column is datetime
        data_temporal = data.copy()
        data_temporal[date_col] = pd.to_datetime(data_temporal[date_col])
        
        # Sort by date
        data_temporal = data_temporal.sort_values(date_col)
        
        # Create results dictionary
        results = {
            'temporal_performance': {},
            'time_window_analysis': {},
            'trend_analysis': {},
            'seasonal_analysis': {}
        }
        
        # Analyze performance over time windows
        results['time_window_analysis'] = self._analyze_time_windows(
            y_true, y_pred, y_scores, data_temporal, date_col
        )
        
        # Analyze temporal trends
        results['trend_analysis'] = self._analyze_temporal_trends(
            y_true, y_pred, y_scores, data_temporal, date_col
        )
        
        # Analyze seasonal patterns
        results['seasonal_analysis'] = self._analyze_seasonal_patterns(
            y_true, y_pred, y_scores, data_temporal, date_col
        )
        
        # Overall temporal performance
        results['temporal_performance'] = self._compute_temporal_performance(
            y_true, y_pred, y_scores, data_temporal, date_col
        )
        
        logger.info("Temporal performance analysis completed")
        return results
    
    def _analyze_time_windows(self, y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray,
                            data: pd.DataFrame, date_col: str) -> Dict[str, Any]:
        """
        Analyze performance across different time windows.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_scores: Prediction scores/probabilities
            data: DataFrame containing temporal data
            date_col: Name of the date column
            
        Returns:
            Dictionary containing time window analysis results
        """
        time_window_results = {}
        
        for window_days in self.time_windows:
            logger.info(f"Analyzing time window: {window_days} days")
            
            # Create time windows
            start_date = data[date_col].min()
            end_date = data[date_col].max()
            
            window_results = []
            current_date = start_date
            
            while current_date < end_date:
                window_end = current_date + pd.Timedelta(days=window_days)
                
                # Get data in this window
                window_mask = (data[date_col] >= current_date) & (data[date_col] < window_end)
                
                if window_mask.sum() >= self.min_samples_per_window:
                    y_true_window = y_true[window_mask]
                    y_pred_window = y_pred[window_mask]
                    y_scores_window = y_scores[window_mask]
                    
                    # Compute performance metrics
                    metrics = self._compute_temporal_metrics(
                        y_true_window, y_pred_window, y_scores_window
                    )
                    
                    metrics['window_start'] = current_date.isoformat()
                    metrics['window_end'] = window_end.isoformat()
                    metrics['sample_size'] = len(y_true_window)
                    
                    window_results.append(metrics)
                
                current_date = window_end
            
            time_window_results[f'{window_days}_days'] = window_results
        
        return time_window_results
    
    def _analyze_temporal_trends(self, y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray,
                               data: pd.DataFrame, date_col: str) -> Dict[str, Any]:
        """
        Analyze temporal trends in model performance.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_scores: Prediction scores/probabilities
            data: DataFrame containing temporal data
            date_col: Name of the date column
            
        Returns:
            Dictionary containing temporal trend analysis results
        """
        trend_results = {}
        
        # Create monthly bins
        data_monthly = data.copy()
        data_monthly['year_month'] = data_monthly[date_col].dt.to_period('M')
        
        monthly_metrics = []
        
        for period in data_monthly['year_month'].unique():
            period_mask = data_monthly['year_month'] == period
            
            if period_mask.sum() >= self.min_samples_per_window:
                y_true_period = y_true[period_mask]
                y_pred_period = y_pred[period_mask]
                y_scores_period = y_scores[period_mask]
                
                metrics = self._compute_temporal_metrics(
                    y_true_period, y_pred_period, y_scores_period
                )
                
                metrics['period'] = str(period)
                metrics['sample_size'] = len(y_true_period)
                
                monthly_metrics.append(metrics)
        
        # Analyze trends
        if len(monthly_metrics) >= 3:  # Need at least 3 time points
            trend_results['monthly_performance'] = monthly_metrics
            
            # Compute trend statistics
            trend_results['trend_statistics'] = self._compute_trend_statistics(monthly_metrics)
        
        return trend_results
    
    def _analyze_seasonal_patterns(self, y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray,
                                 data: pd.DataFrame, date_col: str) -> Dict[str, Any]:
        """
        Analyze seasonal patterns in model performance.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_scores: Prediction scores/probabilities
            data: DataFrame containing temporal data
            date_col: Name of the date column
            
        Returns:
            Dictionary containing seasonal analysis results
        """
        seasonal_results = {}
        
        # Extract seasonal information
        data_seasonal = data.copy()
        data_seasonal['month'] = data_seasonal[date_col].dt.month
        data_seasonal['season'] = data_seasonal['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # Analyze by season
        seasonal_metrics = {}
        
        for season in ['Spring', 'Summer', 'Fall', 'Winter']:
            season_mask = data_seasonal['season'] == season
            
            if season_mask.sum() >= self.min_samples_per_window:
                y_true_season = y_true[season_mask]
                y_pred_season = y_pred[season_mask]
                y_scores_season = y_scores[season_mask]
                
                metrics = self._compute_temporal_metrics(
                    y_true_season, y_pred_season, y_scores_season
                )
                
                metrics['sample_size'] = len(y_true_season)
                seasonal_metrics[season] = metrics
        
        seasonal_results['seasonal_performance'] = seasonal_metrics
        
        # Test for seasonal differences
        if len(seasonal_metrics) >= 2:
            seasonal_results['seasonal_tests'] = self._test_seasonal_differences(seasonal_metrics)
        
        return seasonal_results
    
    def _compute_temporal_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                y_scores: np.ndarray) -> Dict[str, float]:
        """
        Compute performance metrics for temporal analysis.
        
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
            if len(np.unique(y_true)) > 1:
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
            # Recall
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
        
        # Additional temporal metrics
        metrics['mean_predicted_risk'] = float(y_scores.mean())
        metrics['std_predicted_risk'] = float(y_scores.std())
        metrics['prevalence'] = float(y_true.mean())
        
        return metrics
    
    def _compute_trend_statistics(self, monthly_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute trend statistics for temporal performance.
        
        Args:
            monthly_metrics: List of monthly performance metrics
            
        Returns:
            Dictionary containing trend statistics
        """
        trend_stats = {}
        
        # Extract time series data
        time_points = list(range(len(monthly_metrics)))
        
        for metric in ['auc_roc', 'accuracy', 'precision', 'recall', 'f1_score']:
            values = [m.get(metric, np.nan) for m in monthly_metrics]
            values = [v for v in values if not np.isnan(v)]
            
            if len(values) >= 3:
                # Compute linear trend
                slope, intercept, r_value, p_value, std_err = stats.linregress(time_points[:len(values)], values)
                
                trend_stats[metric] = {
                    'slope': float(slope),
                    'intercept': float(intercept),
                    'r_squared': float(r_value ** 2),
                    'p_value': float(p_value),
                    'std_error': float(std_err),
                    'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
                }
        
        return trend_stats
    
    def _test_seasonal_differences(self, seasonal_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Test for seasonal differences in performance.
        
        Args:
            seasonal_metrics: Dictionary containing seasonal performance metrics
            
        Returns:
            Dictionary containing seasonal test results
        """
        seasonal_tests = {}
        
        for metric in ['auc_roc', 'accuracy', 'precision', 'recall', 'f1_score']:
            # Collect metric values for each season
            metric_values = []
            season_names = []
            
            for season, metrics in seasonal_metrics.items():
                if metric in metrics and not np.isnan(metrics[metric]):
                    metric_values.append(metrics[metric])
                    season_names.append(season)
            
            if len(metric_values) >= 2:
                # Perform ANOVA test
                try:
                    f_statistic, p_value = stats.f_oneway(*[metric_values])
                    seasonal_tests[metric] = {
                        'f_statistic': float(f_statistic),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05,
                        'seasonal_values': dict(zip(season_names, metric_values))
                    }
                except Exception as e:
                    logger.warning(f"Could not perform seasonal ANOVA test for {metric}: {e}")
                    seasonal_tests[metric] = {
                        'f_statistic': np.nan,
                        'p_value': np.nan,
                        'significant': False,
                        'seasonal_values': dict(zip(season_names, metric_values))
                    }
        
        return seasonal_tests
    
    def _compute_temporal_performance(self, y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray,
                                    data: pd.DataFrame, date_col: str) -> Dict[str, Any]:
        """
        Compute overall temporal performance metrics.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_scores: Prediction scores/probabilities
            data: DataFrame containing temporal data
            date_col: Name of the date column
            
        Returns:
            Dictionary containing temporal performance metrics
        """
        temporal_performance = {}
        
        # Overall performance
        temporal_performance['overall'] = self._compute_temporal_metrics(y_true, y_pred, y_scores)
        
        # Time span information
        temporal_performance['time_span'] = {
            'start_date': data[date_col].min().isoformat(),
            'end_date': data[date_col].max().isoformat(),
            'total_days': (data[date_col].max() - data[date_col].min()).days,
            'total_samples': len(y_true)
        }
        
        # Performance stability
        temporal_performance['stability'] = self._assess_performance_stability(
            y_true, y_pred, y_scores, data, date_col
        )
        
        return temporal_performance
    
    def _assess_performance_stability(self, y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray,
                                    data: pd.DataFrame, date_col: str) -> Dict[str, Any]:
        """
        Assess performance stability over time.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_scores: Prediction scores/probabilities
            data: DataFrame containing temporal data
            date_col: Name of the date column
            
        Returns:
            Dictionary containing stability assessment
        """
        stability = {}
        
        # Create quarterly bins
        data_quarterly = data.copy()
        data_quarterly['quarter'] = data_quarterly[date_col].dt.to_period('Q')
        
        quarterly_metrics = []
        
        for quarter in data_quarterly['quarter'].unique():
            quarter_mask = data_quarterly['quarter'] == quarter
            
            if quarter_mask.sum() >= self.min_samples_per_window:
                y_true_quarter = y_true[quarter_mask]
                y_pred_quarter = y_pred[quarter_mask]
                y_scores_quarter = y_scores[quarter_mask]
                
                metrics = self._compute_temporal_metrics(
                    y_true_quarter, y_pred_quarter, y_scores_quarter
                )
                
                quarterly_metrics.append(metrics)
        
        if len(quarterly_metrics) >= 2:
            # Compute stability metrics
            for metric in ['auc_roc', 'accuracy', 'precision', 'recall', 'f1_score']:
                values = [m.get(metric, np.nan) for m in quarterly_metrics]
                values = [v for v in values if not np.isnan(v)]
                
                if len(values) >= 2:
                    stability[metric] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'cv': float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else np.inf,
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'range': float(np.max(values) - np.min(values))
                    }
        
        return stability
    
    def perform_reidentification_bias_analysis(self, data: pd.DataFrame, 
                                             patient_id_col: str = 'patient_id',
                                             date_col: str = 'date') -> Dict[str, Any]:
        """
        Perform re-identification bias analysis.
        
        Args:
            data: DataFrame containing temporal data
            patient_id_col: Name of the patient ID column
            date_col: Name of the date column
            
        Returns:
            Dictionary containing re-identification bias analysis results
        """
        logger.info("Performing re-identification bias analysis")
        
        # Ensure date column is datetime
        data_bias = data.copy()
        data_bias[date_col] = pd.to_datetime(data_bias[date_col])
        
        # Sort by patient and date
        data_bias = data_bias.sort_values([patient_id_col, date_col])
        
        bias_results = {
            'patient_frequency_analysis': {},
            'temporal_pattern_analysis': {},
            'data_leakage_analysis': {},
            'bias_assessment': {}
        }
        
        # Analyze patient frequency
        bias_results['patient_frequency_analysis'] = self._analyze_patient_frequency(
            data_bias, patient_id_col
        )
        
        # Analyze temporal patterns
        bias_results['temporal_pattern_analysis'] = self._analyze_temporal_patterns(
            data_bias, patient_id_col, date_col
        )
        
        # Analyze data leakage
        bias_results['data_leakage_analysis'] = self._analyze_data_leakage(
            data_bias, patient_id_col, date_col
        )
        
        # Overall bias assessment
        bias_results['bias_assessment'] = self._assess_overall_bias(bias_results)
        
        logger.info("Re-identification bias analysis completed")
        return bias_results
    
    def _analyze_patient_frequency(self, data: pd.DataFrame, patient_id_col: str) -> Dict[str, Any]:
        """
        Analyze patient frequency patterns.
        
        Args:
            data: DataFrame containing temporal data
            patient_id_col: Name of the patient ID column
            
        Returns:
            Dictionary containing patient frequency analysis
        """
        patient_counts = data[patient_id_col].value_counts()
        
        frequency_analysis = {
            'total_patients': len(patient_counts),
            'total_observations': len(data),
            'observations_per_patient': {
                'mean': float(patient_counts.mean()),
                'std': float(patient_counts.std()),
                'min': int(patient_counts.min()),
                'max': int(patient_counts.max()),
                'median': float(patient_counts.median())
            },
            'frequency_distribution': patient_counts.value_counts().sort_index().to_dict()
        }
        
        # Identify potential bias indicators
        high_frequency_patients = patient_counts[patient_counts > patient_counts.quantile(0.95)]
        frequency_analysis['high_frequency_patients'] = {
            'count': len(high_frequency_patients),
            'percentage': len(high_frequency_patients) / len(patient_counts) * 100,
            'patient_ids': high_frequency_patients.index.tolist()
        }
        
        return frequency_analysis
    
    def _analyze_temporal_patterns(self, data: pd.DataFrame, patient_id_col: str, 
                                 date_col: str) -> Dict[str, Any]:
        """
        Analyze temporal patterns for bias detection.
        
        Args:
            data: DataFrame containing temporal data
            patient_id_col: Name of the patient ID column
            date_col: Name of the date column
            
        Returns:
            Dictionary containing temporal pattern analysis
        """
        temporal_patterns = {}
        
        # Analyze time gaps between observations
        time_gaps = []
        
        for patient_id in data[patient_id_col].unique():
            patient_data = data[data[patient_id_col] == patient_id].sort_values(date_col)
            
            if len(patient_data) > 1:
                gaps = patient_data[date_col].diff().dt.days.dropna()
                time_gaps.extend(gaps.tolist())
        
        if time_gaps:
            temporal_patterns['time_gaps'] = {
                'mean_days': float(np.mean(time_gaps)),
                'std_days': float(np.std(time_gaps)),
                'median_days': float(np.median(time_gaps)),
                'min_days': float(np.min(time_gaps)),
                'max_days': float(np.max(time_gaps))
            }
        
        # Analyze observation patterns
        data['year'] = data[date_col].dt.year
        data['month'] = data[date_col].dt.month
        
        temporal_patterns['yearly_distribution'] = data['year'].value_counts().sort_index().to_dict()
        temporal_patterns['monthly_distribution'] = data['month'].value_counts().sort_index().to_dict()
        
        return temporal_patterns
    
    def _analyze_data_leakage(self, data: pd.DataFrame, patient_id_col: str, 
                            date_col: str) -> Dict[str, Any]:
        """
        Analyze potential data leakage.
        
        Args:
            data: DataFrame containing temporal data
            patient_id_col: Name of the patient ID column
            date_col: Name of the date column
            
        Returns:
            Dictionary containing data leakage analysis
        """
        leakage_analysis = {}
        
        # Check for future information leakage
        # This would need to be customized based on the specific use case
        # For now, we'll check for basic temporal consistency
        
        # Analyze patient observation patterns
        patient_observation_dates = {}
        
        for patient_id in data[patient_id_col].unique():
            patient_data = data[data[patient_id_col] == patient_id].sort_values(date_col)
            patient_observation_dates[patient_id] = patient_data[date_col].tolist()
        
        # Check for overlapping time periods
        overlapping_patients = 0
        total_patients = len(patient_observation_dates)
        
        for i, (patient1, dates1) in enumerate(patient_observation_dates.items()):
            for j, (patient2, dates2) in enumerate(patient_observation_dates.items()):
                if i < j:  # Avoid duplicate comparisons
                    # Check if there's any temporal overlap
                    if (min(dates1) <= max(dates2)) and (min(dates2) <= max(dates1)):
                        overlapping_patients += 1
        
        leakage_analysis['temporal_overlap'] = {
            'overlapping_patient_pairs': overlapping_patients,
            'total_patient_pairs': total_patients * (total_patients - 1) // 2,
            'overlap_percentage': overlapping_patients / (total_patients * (total_patients - 1) // 2) * 100
        }
        
        return leakage_analysis
    
    def _assess_overall_bias(self, bias_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess overall bias based on analysis results.
        
        Args:
            bias_results: Dictionary containing bias analysis results
            
        Returns:
            Dictionary containing overall bias assessment
        """
        bias_assessment = {
            'bias_indicators': [],
            'risk_level': 'low',
            'recommendations': []
        }
        
        # Check patient frequency bias
        frequency_analysis = bias_results.get('patient_frequency_analysis', {})
        high_freq_patients = frequency_analysis.get('high_frequency_patients', {})
        
        if high_freq_patients.get('percentage', 0) > 10:
            bias_assessment['bias_indicators'].append('High frequency patients detected')
            bias_assessment['risk_level'] = 'medium'
            bias_assessment['recommendations'].append(
                'Consider stratified sampling to balance patient representation'
            )
        
        # Check temporal overlap
        leakage_analysis = bias_results.get('data_leakage_analysis', {})
        temporal_overlap = leakage_analysis.get('temporal_overlap', {})
        
        if temporal_overlap.get('overlap_percentage', 0) > 50:
            bias_assessment['bias_indicators'].append('High temporal overlap detected')
            bias_assessment['risk_level'] = 'high'
            bias_assessment['recommendations'].append(
                'Implement strict temporal splitting to prevent data leakage'
            )
        
        # Overall assessment
        if len(bias_assessment['bias_indicators']) == 0:
            bias_assessment['risk_level'] = 'low'
            bias_assessment['recommendations'].append('No significant bias detected')
        
        return bias_assessment
    
    def generate_temporal_report(self, temporal_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive temporal validation report.
        
        Args:
            temporal_results: Dictionary containing temporal validation results
            
        Returns:
            Dictionary containing formatted temporal report
        """
        logger.info("Generating temporal validation report")
        
        report = {
            'summary': {
                'analysis_date': pd.Timestamp.now().isoformat(),
                'time_windows_analyzed': self.time_windows,
                'min_samples_per_window': self.min_samples_per_window
            },
            'temporal_performance': temporal_results.get('temporal_performance', {}),
            'time_window_analysis': temporal_results.get('time_window_analysis', {}),
            'trend_analysis': temporal_results.get('trend_analysis', {}),
            'seasonal_analysis': temporal_results.get('seasonal_analysis', {}),
            'reidentification_bias': temporal_results.get('reidentification_bias', {}),
            'interpretations': self._interpret_temporal_results(temporal_results)
        }
        
        logger.info("Temporal validation report generated")
        return report
    
    def _interpret_temporal_results(self, temporal_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Interpret temporal validation results.
        
        Args:
            temporal_results: Dictionary containing temporal validation results
            
        Returns:
            Dictionary containing interpretations
        """
        interpretations = {}
        
        # Interpret trend analysis
        trend_analysis = temporal_results.get('trend_analysis', {})
        trend_stats = trend_analysis.get('trend_statistics', {})
        
        if trend_stats:
            significant_trends = []
            for metric, stats in trend_stats.items():
                if stats.get('p_value', 1) < 0.05:
                    direction = stats.get('trend_direction', 'stable')
                    significant_trends.append(f"{metric}: {direction}")
            
            if significant_trends:
                interpretations['temporal_trends'] = f"Significant trends: {', '.join(significant_trends)}"
            else:
                interpretations['temporal_trends'] = "No significant temporal trends detected"
        
        # Interpret seasonal analysis
        seasonal_analysis = temporal_results.get('seasonal_analysis', {})
        seasonal_tests = seasonal_analysis.get('seasonal_tests', {})
        
        if seasonal_tests:
            significant_seasons = []
            for metric, test in seasonal_tests.items():
                if test.get('significant', False):
                    significant_seasons.append(metric)
            
            if significant_seasons:
                interpretations['seasonal_patterns'] = f"Significant seasonal differences: {', '.join(significant_seasons)}"
            else:
                interpretations['seasonal_patterns'] = "No significant seasonal patterns detected"
        
        # Interpret bias analysis
        reidentification_bias = temporal_results.get('reidentification_bias', {})
        bias_assessment = reidentification_bias.get('bias_assessment', {})
        
        if bias_assessment:
            risk_level = bias_assessment.get('risk_level', 'unknown')
            interpretations['bias_risk'] = f"Re-identification bias risk: {risk_level}"
            
            recommendations = bias_assessment.get('recommendations', [])
            if recommendations:
                interpretations['bias_recommendations'] = f"Recommendations: {'; '.join(recommendations)}"
        
        return interpretations

