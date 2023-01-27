"""
Validation plots module for cardiovascular risk prediction pipeline.

This module provides comprehensive visualization capabilities for model validation
results including ROC curves, calibration plots, and performance metrics.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import warnings

logger = logging.getLogger(__name__)


class ValidationPlots:
    """
    Provides validation result visualization capabilities.
    
    Specialized for cardiovascular risk prediction model validation with
    support for ROC curves, calibration plots, and performance metrics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize validation plots with configuration.
        
        Args:
            config: Configuration dictionary containing visualization settings
        """
        self.config = config
        self.viz_config = config.get('visualization', {})
        self.plot_config = self.viz_config.get('plots', {})
        self.colors = self.viz_config.get('colors', {})
        
        # Set up matplotlib style
        plt.style.use(self.plot_config.get('style', 'seaborn-v0_8'))
        
        # Set default figure size
        self.figsize = tuple(self.plot_config.get('figure_size', [12, 8]))
        self.dpi = self.plot_config.get('dpi', 300)
        
        logger.info("Validation plots initialized")
    
    def plot_roc_curves(self, y_true: np.ndarray, y_scores: Dict[str, np.ndarray],
                       output_path: Optional[str] = None) -> None:
        """
        Plot ROC curves for multiple models or cross-validation folds.
        
        Args:
            y_true: True binary labels
            y_scores: Dictionary mapping model names to prediction scores
            output_path: Optional path to save the plot
        """
        plt.figure(figsize=self.figsize)
        
        # Plot ROC curve for each model
        for model_name, scores in y_scores.items():
            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
        
        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.500)', alpha=0.7)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('ROC Curves - Cardiovascular Risk Prediction')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"ROC curves plot saved to {output_path}")
        
        plt.show()
    
    def plot_precision_recall_curves(self, y_true: np.ndarray, y_scores: Dict[str, np.ndarray],
                                   output_path: Optional[str] = None) -> None:
        """
        Plot precision-recall curves for multiple models.
        
        Args:
            y_true: True binary labels
            y_scores: Dictionary mapping model names to prediction scores
            output_path: Optional path to save the plot
        """
        plt.figure(figsize=self.figsize)
        
        # Plot PR curve for each model
        for model_name, scores in y_scores.items():
            precision, recall, _ = precision_recall_curve(y_true, scores)
            pr_auc = auc(recall, precision)
            
            plt.plot(recall, precision, label=f'{model_name} (PR-AUC = {pr_auc:.3f})', linewidth=2)
        
        # Plot baseline (random classifier)
        baseline = np.sum(y_true) / len(y_true)
        plt.axhline(y=baseline, color='k', linestyle='--', 
                   label=f'Random Classifier (PR-AUC = {baseline:.3f})', alpha=0.7)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (Sensitivity)')
        plt.ylabel('Precision (PPV)')
        plt.title('Precision-Recall Curves - Cardiovascular Risk Prediction')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Precision-recall curves plot saved to {output_path}")
        
        plt.show()
    
    def plot_confusion_matrices(self, y_true: np.ndarray, y_pred: Dict[str, np.ndarray],
                              output_path: Optional[str] = None) -> None:
        """
        Plot confusion matrices for multiple models.
        
        Args:
            y_true: True binary labels
            y_pred: Dictionary mapping model names to predictions
            output_path: Optional path to save the plot
        """
        n_models = len(y_pred)
        n_cols = min(3, n_models)
        n_rows = (n_models + 2) // 3
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, (model_name, predictions) in enumerate(y_pred.items()):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, predictions)
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{model_name}\nConfusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            
            # Add performance metrics
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            metrics_text = f'Acc: {accuracy:.3f}\nSens: {sensitivity:.3f}\nSpec: {specificity:.3f}\nPPV: {ppv:.3f}\nNPV: {npv:.3f}'
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Hide unused subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Confusion matrices plot saved to {output_path}")
        
        plt.show()
    
    def plot_calibration_curves(self, y_true: np.ndarray, y_scores: Dict[str, np.ndarray],
                              n_bins: int = 10, output_path: Optional[str] = None) -> None:
        """
        Plot calibration curves for multiple models.
        
        Args:
            y_true: True binary labels
            y_scores: Dictionary mapping model names to prediction scores
            n_bins: Number of bins for calibration
            output_path: Optional path to save the plot
        """
        plt.figure(figsize=self.figsize)
        
        # Plot calibration curve for each model
        for model_name, scores in y_scores.items():
            # Calculate calibration curve
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            bin_centers = []
            bin_means = []
            bin_counts = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (scores > bin_lower) & (scores <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    bin_centers.append((bin_lower + bin_upper) / 2)
                    bin_means.append(y_true[in_bin].mean())
                    bin_counts.append(in_bin.sum())
            
            # Plot calibration curve
            plt.plot(bin_centers, bin_means, 'o-', label=f'{model_name}', linewidth=2, markersize=6)
            
            # Calculate calibration slope and intercept
            if len(bin_centers) > 1:
                slope, intercept = np.polyfit(bin_centers, bin_means, 1)
                plt.plot([0, 1], [intercept, slope + intercept], '--', alpha=0.7)
        
        # Plot perfect calibration line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.7)
        
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curves - Cardiovascular Risk Prediction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Calibration curves plot saved to {output_path}")
        
        plt.show()
    
    def plot_performance_metrics(self, metrics: Dict[str, Dict[str, float]],
                               output_path: Optional[str] = None) -> None:
        """
        Plot performance metrics comparison across models.
        
        Args:
            metrics: Dictionary mapping model names to metric dictionaries
            output_path: Optional path to save the plot
        """
        # Extract metrics
        metric_names = list(next(iter(metrics.values())).keys())
        model_names = list(metrics.keys())
        
        # Create subplots
        n_metrics = len(metric_names)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + 2) // 3
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, metric_name in enumerate(metric_names):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            # Extract metric values
            metric_values = [metrics[model][metric_name] for model in model_names]
            
            # Create bar plot
            bars = ax.bar(model_names, metric_values, 
                         color=[self.colors.get('primary', 'blue'),
                               self.colors.get('secondary', 'red'),
                               self.colors.get('success', 'green'),
                               self.colors.get('warning', 'orange')])
            
            ax.set_title(f'{metric_name.upper()}')
            ax.set_ylabel(metric_name.title())
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Performance metrics plot saved to {output_path}")
        
        plt.show()
    
    def plot_subgroup_analysis(self, subgroup_results: Dict[str, Dict[str, Any]],
                             output_path: Optional[str] = None) -> None:
        """
        Plot subgroup analysis results.
        
        Args:
            subgroup_results: Dictionary mapping subgroup names to results
            output_path: Optional path to save the plot
        """
        # Extract subgroup names and metrics
        subgroup_names = list(subgroup_results.keys())
        metrics = list(next(iter(subgroup_results.values())).keys())
        
        # Create subplots
        n_metrics = len(metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            # Extract metric values for each subgroup
            metric_values = []
            metric_errors = []
            
            for subgroup in subgroup_names:
                if metric in subgroup_results[subgroup]:
                    value = subgroup_results[subgroup][metric]
                    if isinstance(value, dict) and 'mean' in value and 'std' in value:
                        metric_values.append(value['mean'])
                        metric_errors.append(value['std'])
                    else:
                        metric_values.append(value)
                        metric_errors.append(0)
                else:
                    metric_values.append(0)
                    metric_errors.append(0)
            
            # Create bar plot with error bars
            bars = ax.bar(subgroup_names, metric_values, 
                         yerr=metric_errors, capsize=5,
                         color=[self.colors.get('primary', 'blue'),
                               self.colors.get('secondary', 'red'),
                               self.colors.get('success', 'green'),
                               self.colors.get('warning', 'orange')])
            
            ax.set_title(f'{metric.upper()} by Subgroup')
            ax.set_ylabel(metric.title())
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Subgroup analysis plot saved to {output_path}")
        
        plt.show()
    
    def plot_temporal_performance(self, temporal_results: Dict[str, Any],
                                output_path: Optional[str] = None) -> None:
        """
        Plot temporal performance analysis.
        
        Args:
            temporal_results: Dictionary containing temporal performance data
            output_path: Optional path to save the plot
        """
        plt.figure(figsize=self.figsize)
        
        # Extract time points and metrics
        time_points = temporal_results.get('time_points', [])
        metrics = temporal_results.get('metrics', {})
        
        # Plot each metric over time
        for metric_name, values in metrics.items():
            plt.plot(time_points, values, 'o-', label=metric_name, linewidth=2, markersize=6)
        
        plt.xlabel('Time Point')
        plt.ylabel('Performance Metric')
        plt.title('Temporal Performance Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Temporal performance plot saved to {output_path}")
        
        plt.show()
    
    def create_validation_report(self, validation_results: Dict[str, Any], output_dir: str) -> None:
        """
        Create comprehensive validation report with all plots.
        
        Args:
            validation_results: Dictionary containing validation results
            output_dir: Directory to save the validation report
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating comprehensive validation report in {output_dir}")
        
        # Generate all validation plots
        try:
            if 'roc_curves' in validation_results:
                self.plot_roc_curves(
                    validation_results['y_true'],
                    validation_results['roc_curves'],
                    output_path=str(output_path / 'roc_curves.png')
                )
        except Exception as e:
            logger.warning(f"Failed to generate ROC curves: {e}")
        
        try:
            if 'precision_recall_curves' in validation_results:
                self.plot_precision_recall_curves(
                    validation_results['y_true'],
                    validation_results['precision_recall_curves'],
                    output_path=str(output_path / 'precision_recall_curves.png')
                )
        except Exception as e:
            logger.warning(f"Failed to generate precision-recall curves: {e}")
        
        try:
            if 'confusion_matrices' in validation_results:
                self.plot_confusion_matrices(
                    validation_results['y_true'],
                    validation_results['confusion_matrices'],
                    output_path=str(output_path / 'confusion_matrices.png')
                )
        except Exception as e:
            logger.warning(f"Failed to generate confusion matrices: {e}")
        
        try:
            if 'calibration_curves' in validation_results:
                self.plot_calibration_curves(
                    validation_results['y_true'],
                    validation_results['calibration_curves'],
                    output_path=str(output_path / 'calibration_curves.png')
                )
        except Exception as e:
            logger.warning(f"Failed to generate calibration curves: {e}")
        
        try:
            if 'performance_metrics' in validation_results:
                self.plot_performance_metrics(
                    validation_results['performance_metrics'],
                    output_path=str(output_path / 'performance_metrics.png')
                )
        except Exception as e:
            logger.warning(f"Failed to generate performance metrics: {e}")
        
        try:
            if 'subgroup_analysis' in validation_results:
                self.plot_subgroup_analysis(
                    validation_results['subgroup_analysis'],
                    output_path=str(output_path / 'subgroup_analysis.png')
                )
        except Exception as e:
            logger.warning(f"Failed to generate subgroup analysis: {e}")
        
        try:
            if 'temporal_performance' in validation_results:
                self.plot_temporal_performance(
                    validation_results['temporal_performance'],
                    output_path=str(output_path / 'temporal_performance.png')
                )
        except Exception as e:
            logger.warning(f"Failed to generate temporal performance: {e}")
        
        logger.info("Validation report generated successfully")
