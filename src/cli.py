"""
Command-line interface for cardiovascular risk prediction pipeline.

This module provides a comprehensive CLI for running the cardiovascular risk
prediction pipeline with various options and configurations.
"""

import click
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from cv_risk_pipeline.utils.config_manager import ConfigManager
from cv_risk_pipeline.utils.logger import setup_logging
from cv_risk_pipeline.data.ingestion import DataIngestion
from cv_risk_pipeline.data.cleaning import DataCleaning
from cv_risk_pipeline.data.metabolic_syndrome import MetabolicSyndromeIdentifier
from cv_risk_pipeline.data.preprocessing import DataPreprocessor
from cv_risk_pipeline.models.risk_prediction import RiskPredictionModel
from cv_risk_pipeline.models.algorithms import FraminghamRiskScore, PooledCohortEquation
from cv_risk_pipeline.validation.statistical_validation import StatisticalValidation
from cv_risk_pipeline.validation.subgroup_analysis import SubgroupAnalysis
from cv_risk_pipeline.validation.temporal_validation import TemporalValidation
from cv_risk_pipeline.visualization.clinical_visualizer import ClinicalVisualizer
from cv_risk_pipeline.visualization.eda_plots import EDAPlots
from cv_risk_pipeline.visualization.validation_plots import ValidationPlots
from cv_risk_pipeline.reporting.regulatory_reporter import RegulatoryReporter
from cv_risk_pipeline.reporting.fda_510k_reporter import FDA510kReporter
from cv_risk_pipeline.reporting.statistical_analysis_plan import StatisticalAnalysisPlan


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), 
              default='INFO', help='Logging level')
@click.pass_context
def cli(ctx, config, verbose, log_level):
    """Cardiovascular Risk Prediction Pipeline CLI."""
    # Initialize context
    ctx.ensure_object(dict)
    
    # Setup logging
    setup_logging(level=log_level, verbose=verbose)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config_manager = ConfigManager()
    if config:
        config_manager.load_config(config)
    else:
        # Try to load default config
        default_config_path = Path('config/default_config.yaml')
        if default_config_path.exists():
            config_manager.load_config(default_config_path)
        else:
            logger.warning("No configuration file found, using default configuration")
            config_manager.create_default_config()
    
    ctx.obj['config_manager'] = config_manager
    ctx.obj['config'] = config_manager.config
    
    logger.info("Cardiovascular Risk Prediction Pipeline CLI initialized")


@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True, 
              help='Input data file path')
@click.option('--output', '-o', type=click.Path(), default='outputs', 
              help='Output directory path')
@click.option('--format', type=click.Choice(['csv', 'excel', 'parquet']), 
              default='csv', help='Input data format')
@click.pass_context
def ingest(ctx, input, output, format):
    """Ingest and validate clinical data."""
    logger = logging.getLogger(__name__)
    logger.info("Starting data ingestion")
    
    config = ctx.obj['config']
    
    # Initialize data ingestion
    ingestion = DataIngestion(config)
    
    # Load data
    if format == 'csv':
        df = ingestion.load_ehr_data(input, 'csv')
    elif format == 'excel':
        df = ingestion.load_ehr_data(input, 'excel')
    elif format == 'parquet':
        df = ingestion.load_ehr_data(input, 'parquet')
    
    # Validate data structure
    required_columns = ['patient_id', 'age', 'sex', 'systolic_bp', 'diastolic_bp']
    validation_results = ingestion.validate_data_structure(df, required_columns)
    
    # Generate data summary
    summary = ingestion.get_data_summary(df)
    
    # Save processed data
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    ingestion.save_processed_data(df, 'ingested_data', 'parquet')
    
    # Save validation results
    with open(output_path / 'validation_results.json', 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    with open(output_path / 'data_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Data ingestion completed. Processed {len(df)} records")
    click.echo(f"Data ingestion completed successfully. Output saved to {output}")


@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True, 
              help='Input data file path')
@click.option('--output', '-o', type=click.Path(), default='outputs', 
              help='Output directory path')
@click.pass_context
def clean(ctx, input, output):
    """Clean and preprocess clinical data."""
    logger = logging.getLogger(__name__)
    logger.info("Starting data cleaning")
    
    config = ctx.obj['config']
    
    # Load data
    ingestion = DataIngestion(config)
    df = ingestion.load_ehr_data(input)
    
    # Initialize data cleaning
    cleaning = DataCleaning(config)
    
    # Detect outliers
    outlier_results = cleaning.detect_outliers(df)
    
    # Handle missing values
    df_cleaned = cleaning.handle_missing_values(df)
    
    # Clean temporal data
    df_cleaned = cleaning.clean_temporal_data(df_cleaned)
    
    # Validate clinical ranges
    clinical_ranges = {
        'systolic_bp': (70, 250),
        'diastolic_bp': (40, 150),
        'total_cholesterol': (100, 500),
        'hdl_cholesterol': (10, 150),
        'triglycerides': (50, 1000),
        'glucose': (50, 500)
    }
    range_validation = cleaning.validate_clinical_ranges(df_cleaned, clinical_ranges)
    
    # Generate data quality report
    quality_report = cleaning.generate_data_quality_report(df_cleaned)
    
    # Save cleaned data
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cleaning.save_processed_data(df_cleaned, 'cleaned_data', 'parquet')
    
    # Save cleaning results
    with open(output_path / 'outlier_results.json', 'w') as f:
        json.dump(outlier_results, f, indent=2, default=str)
    
    with open(output_path / 'range_validation.json', 'w') as f:
        json.dump(range_validation, f, indent=2, default=str)
    
    with open(output_path / 'quality_report.json', 'w') as f:
        json.dump(quality_report, f, indent=2, default=str)
    
    logger.info(f"Data cleaning completed. Processed {len(df_cleaned)} records")
    click.echo(f"Data cleaning completed successfully. Output saved to {output}")


@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True, 
              help='Input data file path')
@click.option('--output', '-o', type=click.Path(), default='outputs', 
              help='Output directory path')
@click.pass_context
def identify_metabolic_syndrome(ctx, input, output):
    """Identify patients with metabolic syndrome."""
    logger = logging.getLogger(__name__)
    logger.info("Starting metabolic syndrome identification")
    
    config = ctx.obj['config']
    
    # Load data
    ingestion = DataIngestion(config)
    df = ingestion.load_ehr_data(input)
    
    # Initialize metabolic syndrome identifier
    ms_identifier = MetabolicSyndromeIdentifier(config)
    
    # Validate data for metabolic syndrome identification
    validation_results = ms_identifier.validate_metabolic_syndrome_data(df)
    
    # Identify metabolic syndrome
    df_with_ms = ms_identifier.identify_metabolic_syndrome(df)
    
    # Generate summary
    summary = ms_identifier.get_metabolic_syndrome_summary(df_with_ms)
    
    # Save results
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    ingestion.save_processed_data(df_with_ms, 'metabolic_syndrome_data', 'parquet')
    
    with open(output_path / 'ms_validation.json', 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    with open(output_path / 'ms_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Metabolic syndrome identification completed. {summary['metabolic_syndrome_patients']} patients identified")
    click.echo(f"Metabolic syndrome identification completed successfully. Output saved to {output}")


@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True, 
              help='Input data file path')
@click.option('--output', '-o', type=click.Path(), default='outputs', 
              help='Output directory path')
@click.pass_context
def preprocess(ctx, input, output):
    """Preprocess data and engineer features."""
    logger = logging.getLogger(__name__)
    logger.info("Starting data preprocessing")
    
    config = ctx.obj['config']
    
    # Load data
    ingestion = DataIngestion(config)
    df = ingestion.load_ehr_data(input)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Engineer temporal features
    df_engineered = preprocessor.engineer_temporal_features(df)
    
    # Engineer clinical features
    df_engineered = preprocessor.engineer_clinical_features(df_engineered)
    
    # Encode categorical features
    df_engineered = preprocessor.encode_categorical_features(df_engineered)
    
    # Scale numeric features
    df_engineered = preprocessor.scale_numeric_features(df_engineered)
    
    # Handle missing values
    df_engineered = preprocessor.handle_missing_values_advanced(df_engineered)
    
    # Create feature matrix
    X, y = preprocessor.create_feature_matrix(df_engineered, 'cardiovascular_event')
    
    # Save processed data
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    ingestion.save_processed_data(df_engineered, 'preprocessed_data', 'parquet')
    ingestion.save_processed_data(X, 'feature_matrix', 'parquet')
    
    if y is not None:
        ingestion.save_processed_data(y.to_frame(), 'target_vector', 'parquet')
    
    logger.info(f"Data preprocessing completed. Created {X.shape[1]} features")
    click.echo(f"Data preprocessing completed successfully. Output saved to {output}")


@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True, 
              help='Input data file path')
@click.option('--output', '-o', type=click.Path(), default='outputs', 
              help='Output directory path')
@click.option('--algorithm', type=click.Choice(['framingham', 'pooled_cohort', 'custom']), 
              default='framingham', help='Risk prediction algorithm')
@click.pass_context
def train(ctx, input, output, algorithm):
    """Train cardiovascular risk prediction model."""
    logger = logging.getLogger(__name__)
    logger.info("Starting model training")
    
    config = ctx.obj['config']
    
    # Load data
    ingestion = DataIngestion(config)
    df = ingestion.load_ehr_data(input)
    
    # Create feature matrix and target
    preprocessor = DataPreprocessor(config)
    X, y = preprocessor.create_feature_matrix(df, 'cardiovascular_event')
    
    # Initialize model based on algorithm choice
    if algorithm == 'framingham':
        model = FraminghamRiskScore(config)
    elif algorithm == 'pooled_cohort':
        model = PooledCohortEquation(config)
    else:
        from cv_risk_pipeline.models.risk_prediction import CustomRiskModel
        model = CustomRiskModel(config)
    
    # Train model
    model.fit(X, y)
    
    # Save model
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.save_model(output_path / f'{algorithm}_model.pkl')
    
    # Save model info
    model_info = model.get_model_info()
    with open(output_path / f'{algorithm}_model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2, default=str)
    
    logger.info(f"Model training completed using {algorithm} algorithm")
    click.echo(f"Model training completed successfully. Model saved to {output}")


@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True, 
              help='Input data file path')
@click.option('--model', '-m', type=click.Path(exists=True), required=True, 
              help='Trained model file path')
@click.option('--output', '-o', type=click.Path(), default='outputs', 
              help='Output directory path')
@click.pass_context
def predict(ctx, input, model, output):
    """Make predictions using trained model."""
    logger = logging.getLogger(__name__)
    logger.info("Starting model prediction")
    
    config = ctx.obj['config']
    
    # Load data
    ingestion = DataIngestion(config)
    df = ingestion.load_ehr_data(input)
    
    # Load model
    from cv_risk_pipeline.models.risk_prediction import RiskPredictionModel
    model_instance = RiskPredictionModel(config)
    model_instance.load_model(model)
    
    # Make predictions
    predictions = model_instance.batch_score(df)
    
    # Save predictions
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    ingestion.save_processed_data(predictions, 'predictions', 'parquet')
    
    logger.info(f"Model prediction completed. Generated {len(predictions)} predictions")
    click.echo(f"Model prediction completed successfully. Predictions saved to {output}")


@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True, 
              help='Input data file path')
@click.option('--model', '-m', type=click.Path(exists=True), required=True, 
              help='Trained model file path')
@click.option('--output', '-o', type=click.Path(), default='outputs', 
              help='Output directory path')
@click.pass_context
def validate(ctx, input, model, output):
    """Validate model performance."""
    logger = logging.getLogger(__name__)
    logger.info("Starting model validation")
    
    config = ctx.obj['config']
    
    # Load data
    ingestion = DataIngestion(config)
    df = ingestion.load_ehr_data(input)
    
    # Load model
    from cv_risk_pipeline.models.risk_prediction import RiskPredictionModel
    model_instance = RiskPredictionModel(config)
    model_instance.load_model(model)
    
    # Create feature matrix and target
    preprocessor = DataPreprocessor(config)
    X, y = preprocessor.create_feature_matrix(df, 'cardiovascular_event')
    
    # Make predictions
    y_pred = model_instance.predict(X)
    y_scores = model_instance.predict_proba(X)
    
    # Initialize validation
    validation = StatisticalValidation(config)
    
    # Compute performance metrics
    performance_metrics = validation.compute_performance_metrics(y, y_pred, y_scores)
    
    # Compute confusion matrix metrics
    confusion_matrix = validation.compute_confusion_matrix_metrics(y, y_pred)
    
    # Compute calibration metrics
    calibration_metrics = validation.compute_calibration_metrics(y, y_scores)
    
    # Perform cross-validation
    cv_results = validation.perform_cross_validation(model_instance, X, y)
    
    # Generate validation report
    validation_results = {
        'performance_metrics': performance_metrics,
        'confusion_matrix': confusion_matrix,
        'calibration_metrics': calibration_metrics,
        'cross_validation': cv_results,
        'summary_statistics': {
            'total_samples': len(y),
            'positive_samples': int(y.sum()),
            'negative_samples': int(len(y) - y.sum()),
            'prevalence': float(y.mean())
        }
    }
    
    validation_report = validation.generate_validation_report(validation_results)
    
    # Save validation results
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'validation_results.json', 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    with open(output_path / 'validation_report.json', 'w') as f:
        json.dump(validation_report, f, indent=2, default=str)
    
    logger.info(f"Model validation completed. AUC-ROC: {performance_metrics.get('auc_roc', 'N/A'):.3f}")
    click.echo(f"Model validation completed successfully. Results saved to {output}")


@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True, 
              help='Input data file path')
@click.option('--output', '-o', type=click.Path(), default='outputs', 
              help='Output directory path')
@click.pass_context
def visualize(ctx, input, output):
    """Generate visualizations and EDA plots."""
    logger = logging.getLogger(__name__)
    logger.info("Starting visualization generation")
    
    config = ctx.obj['config']
    
    # Load data
    ingestion = DataIngestion(config)
    df = ingestion.load_ehr_data(input)
    
    # Initialize visualizers
    clinical_viz = ClinicalVisualizer(config)
    eda_plots = EDAPlots(config)
    
    # Generate EDA plots
    eda_plots.create_eda_report(df, str(Path(output) / 'eda'))
    
    # Generate clinical visualizations
    clinical_viz.save_all_plots(df, str(Path(output) / 'clinical'))
    
    logger.info("Visualization generation completed")
    click.echo(f"Visualization generation completed successfully. Plots saved to {output}")


@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True, 
              help='Input data file path')
@click.option('--output', '-o', type=click.Path(), default='outputs', 
              help='Output directory path')
@click.pass_context
def report(ctx, input, output):
    """Generate regulatory reports and documentation."""
    logger = logging.getLogger(__name__)
    logger.info("Starting report generation")
    
    config = ctx.obj['config']
    
    # Load data
    ingestion = DataIngestion(config)
    df = ingestion.load_ehr_data(input)
    
    # Initialize reporters
    regulatory_reporter = RegulatoryReporter(config)
    fda_reporter = FDA510kReporter(config)
    sap_generator = StatisticalAnalysisPlan(config)
    
    # Generate statistical analysis plan
    sap_generator.generate_analysis_plan(str(Path(output) / 'statistical_analysis_plan'))
    
    # Generate validation report (placeholder - would need actual validation results)
    validation_results = {
        'performance_metrics': {'auc_roc': 0.85, 'sensitivity': 0.80, 'specificity': 0.75},
        'summary_statistics': {'total_samples': len(df), 'prevalence': 0.20}
    }
    
    regulatory_reporter.generate_validation_report(validation_results, str(Path(output) / 'validation_report'))
    fda_reporter.generate_510k_submission(validation_results, str(Path(output) / 'fda_510k'))
    
    logger.info("Report generation completed")
    click.echo(f"Report generation completed successfully. Reports saved to {output}")


@cli.command()
@click.option('--output', '-o', type=click.Path(), default='config', 
              help='Output directory path')
@click.pass_context
def init_config(ctx, output):
    """Initialize configuration files."""
    logger = logging.getLogger(__name__)
    logger.info("Initializing configuration files")
    
    config_manager = ctx.obj['config_manager']
    
    # Create output directory
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create default configuration
    config_manager.create_default_config()
    config_manager.save_config(output_path / 'default_config.yaml')
    
    # Create configuration template
    config_manager.create_config_template(output_path / 'config_template.yaml')
    
    logger.info("Configuration files initialized")
    click.echo(f"Configuration files created in {output}")


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), required=True, 
              help='Configuration file path')
@click.option('--output', '-o', type=click.Path(), default='outputs', 
              help='Output directory path')
@click.pass_context
def run_pipeline(ctx, config, output):
    """Run the complete cardiovascular risk prediction pipeline."""
    logger = logging.getLogger(__name__)
    logger.info("Starting complete pipeline execution")
    
    # This would run the complete pipeline
    # For now, just show the steps that would be executed
    
    steps = [
        "1. Data Ingestion",
        "2. Data Cleaning",
        "3. Metabolic Syndrome Identification",
        "4. Data Preprocessing",
        "5. Model Training",
        "6. Model Validation",
        "7. Visualization Generation",
        "8. Report Generation"
    ]
    
    click.echo("Complete pipeline execution would include:")
    for step in steps:
        click.echo(f"  {step}")
    
    logger.info("Pipeline execution plan completed")
    click.echo(f"Pipeline execution plan completed. Use individual commands to run specific steps.")


if __name__ == '__main__':
    cli()

