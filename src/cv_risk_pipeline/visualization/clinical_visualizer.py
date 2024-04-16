"""
Clinical data visualization module for cardiovascular risk prediction pipeline.

This module provides specialized visualization capabilities for clinical data,
including longitudinal plots, risk factor distributions, and clinical trends.
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


class ClinicalVisualizer:
    """
    Provides clinical data visualization capabilities.
    
    Specialized for cardiovascular risk prediction with support for
    longitudinal data, clinical trends, and risk factor analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize clinical visualizer with configuration.
        
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
        
        logger.info("Clinical visualizer initialized")
    
    def plot_longitudinal_trajectories(self, df: pd.DataFrame,
                                     patient_id_col: str = 'patient_id',
                                     date_col: str = 'date',
                                     value_cols: List[str] = None,
                                     output_path: Optional[str] = None) -> None:
        """
        Plot longitudinal trajectories for clinical variables.
        
        Args:
            df: DataFrame containing longitudinal data
            patient_id_col: Name of the patient ID column
            date_col: Name of the date column
            value_cols: List of columns to plot
            output_path: Optional path to save the plot
        """
        if value_cols is None:
            value_cols = ['systolic_bp', 'diastolic_bp', 'total_cholesterol', 'hdl_cholesterol']
        
        # Filter to available columns
        available_cols = [col for col in value_cols if col in df.columns]
        if not available_cols:
            logger.warning("No valid columns found for longitudinal plotting")
            return
        
        # Create subplots
        n_cols = min(2, len(available_cols))
        n_rows = (len(available_cols) + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Ensure date column is datetime
        df_plot = df.copy()
        df_plot[date_col] = pd.to_datetime(df_plot[date_col])
        
        # Plot each variable
        for i, col in enumerate(available_cols):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            # Sample patients for plotting (to avoid overcrowding)
            unique_patients = df_plot[patient_id_col].unique()
            if len(unique_patients) > 50:
                sample_patients = np.random.choice(unique_patients, 50, replace=False)
                df_sample = df_plot[df_plot[patient_id_col].isin(sample_patients)]
            else:
                df_sample = df_plot
            
            # Plot individual trajectories
            for patient_id in df_sample[patient_id_col].unique():
                patient_data = df_sample[df_sample[patient_id_col] == patient_id].sort_values(date_col)
                ax.plot(patient_data[date_col], patient_data[col], 
                       alpha=0.3, linewidth=0.5, color='lightblue')
            
            # Plot mean trajectory
            mean_trajectory = df_plot.groupby(date_col)[col].mean()
            ax.plot(mean_trajectory.index, mean_trajectory.values, 
                   color='red', linewidth=2, label='Mean')
            
            ax.set_title(f'Longitudinal {col.replace("_", " ").title()}')
            ax.set_xlabel('Date')
            ax.set_ylabel(col.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(available_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Longitudinal trajectories plot saved to {output_path}")
        
        plt.show()
    
    def plot_risk_factor_distributions(self, df: pd.DataFrame,
                                     risk_factors: List[str] = None,
                                     group_by: Optional[str] = None,
                                     output_path: Optional[str] = None) -> None:
        """
        Plot distributions of cardiovascular risk factors.
        
        Args:
            df: DataFrame containing clinical data
            risk_factors: List of risk factor columns to plot
            group_by: Optional column to group by (e.g., 'metabolic_syndrome')
            output_path: Optional path to save the plot
        """
        if risk_factors is None:
            risk_factors = ['age', 'systolic_bp', 'diastolic_bp', 'total_cholesterol', 
                          'hdl_cholesterol', 'triglycerides', 'glucose', 'bmi']
        
        # Filter to available columns
        available_factors = [col for col in risk_factors if col in df.columns]
        if not available_factors:
            logger.warning("No valid risk factors found for distribution plotting")
            return
        
        # Create subplots
        n_cols = min(3, len(available_factors))
        n_rows = (len(available_factors) + 2) // 3
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Plot each risk factor
        for i, factor in enumerate(available_factors):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            if group_by and group_by in df.columns:
                # Grouped distribution
                for group in df[group_by].unique():
                    group_data = df[df[group_by] == group][factor].dropna()
                    if len(group_data) > 0:
                        ax.hist(group_data, alpha=0.6, label=f'{group_by}: {group}', bins=30)
                ax.legend()
            else:
                # Single distribution
                data = df[factor].dropna()
                ax.hist(data, bins=30, alpha=0.7, color=self.colors.get('primary', 'blue'))
            
            ax.set_title(f'{factor.replace("_", " ").title()} Distribution')
            ax.set_xlabel(factor.replace('_', ' ').title())
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(available_factors), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Risk factor distributions plot saved to {output_path}")
        
        plt.show()
    
    def plot_correlation_heatmap(self, df: pd.DataFrame,
                               numeric_columns: List[str] = None,
                               method: str = 'pearson',
                               output_path: Optional[str] = None) -> None:
        """
        Plot correlation heatmap for clinical variables.
        
        Args:
            df: DataFrame containing clinical data
            numeric_columns: List of numeric columns to include
            method: Correlation method ('pearson', 'spearman', 'kendall')
            output_path: Optional path to save the plot
        """
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter to available columns
        available_cols = [col for col in numeric_columns if col in df.columns]
        if len(available_cols) < 2:
            logger.warning("Need at least 2 numeric columns for correlation heatmap")
            return
        
        # Calculate correlation matrix
        corr_matrix = df[available_cols].corr(method=method)
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title(f'Clinical Variables Correlation Matrix ({method.title()})')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Correlation heatmap saved to {output_path}")
        
        plt.show()
    
    def plot_metabolic_syndrome_analysis(self, df: pd.DataFrame,
                                       output_path: Optional[str] = None) -> None:
        """
        Plot metabolic syndrome analysis and prevalence.
        
        Args:
            df: DataFrame containing metabolic syndrome data
            output_path: Optional path to save the plot
        """
        if 'metabolic_syndrome' not in df.columns:
            logger.warning("Metabolic syndrome column not found in data")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Metabolic syndrome prevalence
        ms_counts = df['metabolic_syndrome'].value_counts()
        axes[0, 0].pie(ms_counts.values, labels=['No MS', 'MS'], autopct='%1.1f%%',
                      colors=[self.colors.get('primary', 'lightblue'), 
                             self.colors.get('secondary', 'lightcoral')])
        axes[0, 0].set_title('Metabolic Syndrome Prevalence')
        
        # 2. Criteria distribution
        if 'ms_criteria_count' in df.columns:
            criteria_dist = df['ms_criteria_count'].value_counts().sort_index()
            axes[0, 1].bar(criteria_dist.index, criteria_dist.values,
                          color=self.colors.get('primary', 'blue'))
            axes[0, 1].set_title('Number of Positive Criteria')
            axes[0, 1].set_xlabel('Criteria Count')
            axes[0, 1].set_ylabel('Number of Patients')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Severity distribution
        if 'ms_severity' in df.columns:
            severity_dist = df['ms_severity'].value_counts()
            axes[1, 0].bar(severity_dist.index, severity_dist.values,
                          color=[self.colors.get('primary', 'blue'),
                                self.colors.get('success', 'green'),
                                self.colors.get('warning', 'orange'),
                                self.colors.get('secondary', 'red')])
            axes[1, 0].set_title('Metabolic Syndrome Severity')
            axes[1, 0].set_xlabel('Severity Level')
            axes[1, 0].set_ylabel('Number of Patients')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Individual criteria prevalence
        criteria_cols = [col for col in df.columns if col.startswith('ms_') and col.endswith('_positive')]
        if criteria_cols:
            criteria_prevalence = []
            criteria_names = []
            for col in criteria_cols:
                criteria_name = col.replace('ms_', '').replace('_positive', '')
                prevalence = df[col].mean() * 100
                criteria_prevalence.append(prevalence)
                criteria_names.append(criteria_name.replace('_', ' ').title())
            
            axes[1, 1].barh(criteria_names, criteria_prevalence,
                           color=self.colors.get('primary', 'blue'))
            axes[1, 1].set_title('Individual Criteria Prevalence (%)')
            axes[1, 1].set_xlabel('Prevalence (%)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Metabolic syndrome analysis plot saved to {output_path}")
        
        plt.show()
    
    def plot_temporal_trends(self, df: pd.DataFrame,
                           patient_id_col: str = 'patient_id',
                           date_col: str = 'date',
                           value_col: str = 'systolic_bp',
                           group_by: Optional[str] = None,
                           output_path: Optional[str] = None) -> None:
        """
        Plot temporal trends for clinical variables.
        
        Args:
            df: DataFrame containing temporal data
            patient_id_col: Name of the patient ID column
            date_col: Name of the date column
            value_col: Name of the value column to plot
            group_by: Optional column to group by
            output_path: Optional path to save the plot
        """
        if value_col not in df.columns:
            logger.warning(f"Value column {value_col} not found in data")
            return
        
        # Ensure date column is datetime
        df_plot = df.copy()
        df_plot[date_col] = pd.to_datetime(df_plot[date_col])
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if group_by and group_by in df_plot.columns:
            # Grouped temporal trends
            for group in df_plot[group_by].unique():
                group_data = df_plot[df_plot[group_by] == group]
                
                # Calculate mean trend over time
                temporal_mean = group_data.groupby(date_col)[value_col].mean()
                temporal_std = group_data.groupby(date_col)[value_col].std()
                
                ax.plot(temporal_mean.index, temporal_mean.values, 
                       label=f'{group_by}: {group}', linewidth=2)
                ax.fill_between(temporal_mean.index, 
                               temporal_mean.values - temporal_std.values,
                               temporal_mean.values + temporal_std.values,
                               alpha=0.2)
        else:
            # Overall temporal trend
            temporal_mean = df_plot.groupby(date_col)[value_col].mean()
            temporal_std = df_plot.groupby(date_col)[value_col].std()
            
            ax.plot(temporal_mean.index, temporal_mean.values, 
                   color=self.colors.get('primary', 'blue'), linewidth=2)
            ax.fill_between(temporal_mean.index, 
                           temporal_mean.values - temporal_std.values,
                           temporal_mean.values + temporal_std.values,
                           alpha=0.3, color=self.colors.get('primary', 'blue'))
        
        ax.set_title(f'Temporal Trend: {value_col.replace("_", " ").title()}')
        ax.set_xlabel('Date')
        ax.set_ylabel(value_col.replace('_', ' ').title())
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Temporal trends plot saved to {output_path}")
        
        plt.show()
    
    def create_interactive_dashboard(self, df: pd.DataFrame,
                                   output_path: Optional[str] = None) -> None:
        """
        Create an interactive dashboard for clinical data exploration.
        
        Args:
            df: DataFrame containing clinical data
            output_path: Optional path to save the dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Risk Factor Distributions', 'Correlation Matrix',
                          'Metabolic Syndrome Analysis', 'Temporal Trends'),
            specs=[[{"type": "histogram"}, {"type": "heatmap"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # 1. Risk factor distributions
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:4]  # Top 4 numeric columns
        for col in numeric_cols:
            fig.add_trace(
                go.Histogram(x=df[col], name=col, opacity=0.7),
                row=1, col=1
            )
        
        # 2. Correlation matrix
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                          colorscale='RdBu', zmid=0),
                row=1, col=2
            )
        
        # 3. Metabolic syndrome analysis
        if 'metabolic_syndrome' in df.columns:
            ms_counts = df['metabolic_syndrome'].value_counts()
            fig.add_trace(
                go.Bar(x=ms_counts.index, y=ms_counts.values, name='Metabolic Syndrome'),
                row=2, col=1
            )
        
        # 4. Temporal trends (if date column exists)
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0 and len(numeric_cols) > 0:
            date_col = date_cols[0]
            value_col = numeric_cols[0]
            temporal_mean = df.groupby(date_col)[value_col].mean()
            fig.add_trace(
                go.Scatter(x=temporal_mean.index, y=temporal_mean.values, 
                          mode='lines', name=f'{value_col} Trend'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Clinical Data Interactive Dashboard",
            showlegend=True,
            height=800
        )
        
        if output_path:
            fig.write_html(output_path)
            logger.info(f"Interactive dashboard saved to {output_path}")
        
        fig.show()
    
    def save_all_plots(self, df: pd.DataFrame, output_dir: str) -> None:
        """
        Generate and save all standard clinical plots.
        
        Args:
            df: DataFrame containing clinical data
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating all clinical plots in {output_dir}")
        
        # Generate all plots
        try:
            self.plot_risk_factor_distributions(df, 
                output_path=str(output_path / 'risk_factor_distributions.png'))
        except Exception as e:
            logger.warning(f"Failed to generate risk factor distributions: {e}")
        
        try:
            self.plot_correlation_heatmap(df, 
                output_path=str(output_path / 'correlation_heatmap.png'))
        except Exception as e:
            logger.warning(f"Failed to generate correlation heatmap: {e}")
        
        try:
            self.plot_metabolic_syndrome_analysis(df, 
                output_path=str(output_path / 'metabolic_syndrome_analysis.png'))
        except Exception as e:
            logger.warning(f"Failed to generate metabolic syndrome analysis: {e}")
        
        try:
            self.create_interactive_dashboard(df, 
                output_path=str(output_path / 'interactive_dashboard.html'))
        except Exception as e:
            logger.warning(f"Failed to generate interactive dashboard: {e}")
        
        logger.info("All clinical plots generated successfully")


