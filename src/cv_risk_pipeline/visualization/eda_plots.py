"""
Exploratory Data Analysis (EDA) plots module for cardiovascular risk prediction pipeline.

This module provides comprehensive EDA visualization capabilities including
missing value analysis, outlier detection, and data quality assessment.
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
import warnings

logger = logging.getLogger(__name__)


class EDAPlots:
    """
    Provides exploratory data analysis visualization capabilities.
    
    Specialized for clinical data with focus on data quality assessment,
    missing value analysis, and outlier detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize EDA plots with configuration.
        
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
        
        logger.info("EDA plots initialized")
    
    def plot_missing_values_analysis(self, df: pd.DataFrame,
                                   output_path: Optional[str] = None) -> None:
        """
        Plot comprehensive missing values analysis.
        
        Args:
            df: DataFrame containing the data
            output_path: Optional path to save the plot
        """
        # Calculate missing values
        missing_data = df.isnull().sum()
        missing_percentage = (missing_data / len(df)) * 100
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Missing values count
        missing_data_sorted = missing_data[missing_data > 0].sort_values(ascending=True)
        if len(missing_data_sorted) > 0:
            axes[0, 0].barh(range(len(missing_data_sorted)), missing_data_sorted.values,
                           color=self.colors.get('warning', 'orange'))
            axes[0, 0].set_yticks(range(len(missing_data_sorted)))
            axes[0, 0].set_yticklabels(missing_data_sorted.index)
            axes[0, 0].set_title('Missing Values Count')
            axes[0, 0].set_xlabel('Number of Missing Values')
        else:
            axes[0, 0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center',
                           transform=axes[0, 0].transAxes, fontsize=14)
            axes[0, 0].set_title('Missing Values Count')
        
        # 2. Missing values percentage
        missing_pct_sorted = missing_percentage[missing_percentage > 0].sort_values(ascending=True)
        if len(missing_pct_sorted) > 0:
            axes[0, 1].barh(range(len(missing_pct_sorted)), missing_pct_sorted.values,
                           color=self.colors.get('secondary', 'red'))
            axes[0, 1].set_yticks(range(len(missing_pct_sorted)))
            axes[0, 1].set_yticklabels(missing_pct_sorted.index)
            axes[0, 1].set_title('Missing Values Percentage')
            axes[0, 1].set_xlabel('Percentage (%)')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center',
                           transform=axes[0, 1].transAxes, fontsize=14)
            axes[0, 1].set_title('Missing Values Percentage')
        
        # 3. Missing values heatmap (sample of data)
        sample_size = min(1000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        missing_matrix = df_sample.isnull()
        
        if missing_matrix.any().any():
            sns.heatmap(missing_matrix.T, cbar=True, yticklabels=True, 
                       cmap='viridis', ax=axes[1, 0])
            axes[1, 0].set_title(f'Missing Values Pattern (Sample of {sample_size} records)')
            axes[1, 0].set_xlabel('Record Index')
            axes[1, 0].set_ylabel('Variables')
        else:
            axes[1, 0].text(0.5, 0.5, 'No Missing Values in Sample', ha='center', va='center',
                           transform=axes[1, 0].transAxes, fontsize=14)
            axes[1, 0].set_title('Missing Values Pattern')
        
        # 4. Data completeness summary
        completeness = ((len(df) - missing_data) / len(df)) * 100
        completeness_sorted = completeness.sort_values(ascending=True)
        
        axes[1, 1].barh(range(len(completeness_sorted)), completeness_sorted.values,
                       color=self.colors.get('primary', 'blue'))
        axes[1, 1].set_yticks(range(len(completeness_sorted)))
        axes[1, 1].set_yticklabels(completeness_sorted.index)
        axes[1, 1].set_title('Data Completeness by Variable')
        axes[1, 1].set_xlabel('Completeness (%)')
        axes[1, 1].axvline(x=80, color='red', linestyle='--', alpha=0.7, label='80% Threshold')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Missing values analysis plot saved to {output_path}")
        
        plt.show()
    
    def plot_outlier_analysis(self, df: pd.DataFrame,
                            numeric_columns: List[str] = None,
                            output_path: Optional[str] = None) -> None:
        """
        Plot comprehensive outlier analysis.
        
        Args:
            df: DataFrame containing the data
            numeric_columns: List of numeric columns to analyze
            output_path: Optional path to save the plot
        """
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter to available columns
        available_cols = [col for col in numeric_columns if col in df.columns]
        if not available_cols:
            logger.warning("No numeric columns found for outlier analysis")
            return
        
        # Limit to top 9 columns for visualization
        available_cols = available_cols[:9]
        
        # Create subplots
        n_cols = 3
        n_rows = (len(available_cols) + 2) // 3
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Plot each column
        for i, col in enumerate(available_cols):
            if i >= len(axes):
                break
            
            ax = axes[i]
            data = df[col].dropna()
            
            if len(data) > 0:
                # Box plot
                box_plot = ax.boxplot(data, patch_artist=True)
                box_plot['boxes'][0].set_facecolor(self.colors.get('primary', 'lightblue'))
                
                # Add outlier information
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = data[(data < lower_bound) | (data > upper_bound)]
                outlier_count = len(outliers)
                outlier_pct = (outlier_count / len(data)) * 100
                
                ax.set_title(f'{col}\nOutliers: {outlier_count} ({outlier_pct:.1f}%)')
                ax.set_ylabel(col.replace('_', ' ').title())
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(col)
        
        # Hide unused subplots
        for i in range(len(available_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Outlier analysis plot saved to {output_path}")
        
        plt.show()
    
    def plot_data_quality_summary(self, df: pd.DataFrame,
                                output_path: Optional[str] = None) -> None:
        """
        Plot comprehensive data quality summary.
        
        Args:
            df: DataFrame containing the data
            output_path: Optional path to save the plot
        """
        # Calculate data quality metrics
        total_records = len(df)
        total_columns = len(df.columns)
        missing_data = df.isnull().sum()
        duplicate_records = df.duplicated().sum()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Data overview
        overview_data = {
            'Total Records': total_records,
            'Total Columns': total_columns,
            'Duplicate Records': duplicate_records,
            'Memory Usage (MB)': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        axes[0, 0].bar(overview_data.keys(), overview_data.values(),
                      color=[self.colors.get('primary', 'blue'),
                            self.colors.get('secondary', 'red'),
                            self.colors.get('warning', 'orange'),
                            self.colors.get('success', 'green')])
        axes[0, 0].set_title('Data Overview')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Data types distribution
        dtype_counts = df.dtypes.value_counts()
        axes[0, 1].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Data Types Distribution')
        
        # 3. Missing values by data type
        missing_by_type = {}
        for dtype in df.dtypes.unique():
            cols_of_type = df.select_dtypes(include=[dtype]).columns
            missing_count = df[cols_of_type].isnull().sum().sum()
            missing_by_type[str(dtype)] = missing_count
        
        if missing_by_type:
            axes[1, 0].bar(missing_by_type.keys(), missing_by_type.values(),
                          color=self.colors.get('warning', 'orange'))
            axes[1, 0].set_title('Missing Values by Data Type')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Data completeness heatmap
        completeness = ((total_records - missing_data) / total_records) * 100
        completeness_sorted = completeness.sort_values(ascending=True)
        
        # Create heatmap data
        heatmap_data = completeness_sorted.values.reshape(-1, 1)
        im = axes[1, 1].imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
        axes[1, 1].set_yticks(range(len(completeness_sorted)))
        axes[1, 1].set_yticklabels(completeness_sorted.index)
        axes[1, 1].set_title('Data Completeness Heatmap')
        axes[1, 1].set_xlabel('Completeness')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1, 1])
        cbar.set_label('Completeness (%)')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Data quality summary plot saved to {output_path}")
        
        plt.show()
    
    def plot_variable_distributions(self, df: pd.DataFrame,
                                  numeric_columns: List[str] = None,
                                  output_path: Optional[str] = None) -> None:
        """
        Plot distributions of all variables.
        
        Args:
            df: DataFrame containing the data
            numeric_columns: List of numeric columns to plot
            output_path: Optional path to save the plot
        """
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter to available columns
        available_cols = [col for col in numeric_columns if col in df.columns]
        if not available_cols:
            logger.warning("No numeric columns found for distribution plotting")
            return
        
        # Limit to top 9 columns for visualization
        available_cols = available_cols[:9]
        
        # Create subplots
        n_cols = 3
        n_rows = (len(available_cols) + 2) // 3
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Plot each column
        for i, col in enumerate(available_cols):
            if i >= len(axes):
                break
            
            ax = axes[i]
            data = df[col].dropna()
            
            if len(data) > 0:
                # Histogram
                ax.hist(data, bins=30, alpha=0.7, color=self.colors.get('primary', 'blue'))
                
                # Add statistics
                mean_val = data.mean()
                median_val = data.median()
                std_val = data.std()
                
                ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
                
                ax.set_title(f'{col}\nMean: {mean_val:.2f}, Std: {std_val:.2f}')
                ax.set_xlabel(col.replace('_', ' ').title())
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(col)
        
        # Hide unused subplots
        for i in range(len(available_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Variable distributions plot saved to {output_path}")
        
        plt.show()
    
    def plot_categorical_analysis(self, df: pd.DataFrame,
                                categorical_columns: List[str] = None,
                                output_path: Optional[str] = None) -> None:
        """
        Plot analysis of categorical variables.
        
        Args:
            df: DataFrame containing the data
            categorical_columns: List of categorical columns to analyze
            output_path: Optional path to save the plot
        """
        if categorical_columns is None:
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Filter to available columns
        available_cols = [col for col in categorical_columns if col in df.columns]
        if not available_cols:
            logger.warning("No categorical columns found for analysis")
            return
        
        # Limit to top 6 columns for visualization
        available_cols = available_cols[:6]
        
        # Create subplots
        n_cols = 2
        n_rows = (len(available_cols) + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Plot each column
        for i, col in enumerate(available_cols):
            if i >= len(axes):
                break
            
            ax = axes[i]
            value_counts = df[col].value_counts()
            
            if len(value_counts) > 0:
                # Bar plot
                bars = ax.bar(range(len(value_counts)), value_counts.values,
                             color=self.colors.get('primary', 'blue'))
                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
                ax.set_title(f'{col}\nUnique Values: {len(value_counts)}')
                ax.set_ylabel('Count')
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, value_counts.values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value}', ha='center', va='bottom')
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(col)
        
        # Hide unused subplots
        for i in range(len(available_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Categorical analysis plot saved to {output_path}")
        
        plt.show()
    
    def create_eda_report(self, df: pd.DataFrame, output_dir: str) -> None:
        """
        Create comprehensive EDA report with all plots.
        
        Args:
            df: DataFrame containing the data
            output_dir: Directory to save the EDA report
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating comprehensive EDA report in {output_dir}")
        
        # Generate all EDA plots
        try:
            self.plot_missing_values_analysis(df, 
                output_path=str(output_path / 'missing_values_analysis.png'))
        except Exception as e:
            logger.warning(f"Failed to generate missing values analysis: {e}")
        
        try:
            self.plot_outlier_analysis(df, 
                output_path=str(output_path / 'outlier_analysis.png'))
        except Exception as e:
            logger.warning(f"Failed to generate outlier analysis: {e}")
        
        try:
            self.plot_data_quality_summary(df, 
                output_path=str(output_path / 'data_quality_summary.png'))
        except Exception as e:
            logger.warning(f"Failed to generate data quality summary: {e}")
        
        try:
            self.plot_variable_distributions(df, 
                output_path=str(output_path / 'variable_distributions.png'))
        except Exception as e:
            logger.warning(f"Failed to generate variable distributions: {e}")
        
        try:
            self.plot_categorical_analysis(df, 
                output_path=str(output_path / 'categorical_analysis.png'))
        except Exception as e:
            logger.warning(f"Failed to generate categorical analysis: {e}")
        
        logger.info("EDA report generated successfully")
