# Cardiovascular Risk Prediction Pipeline

A comprehensive Python-based statistical validation pipeline for clinical decision support systems predicting 10-year cardiovascular risk in patients with metabolic syndrome.

## 🏥 Overview

This pipeline provides a complete solution for developing, validating, and deploying cardiovascular risk prediction models in clinical settings. It handles longitudinal clinical data, implements robust statistical validation, and generates regulatory-compliant documentation for FDA 510(k) submissions.

## ✨ Key Features

- **Robust Data Ingestion**: Handles EHR and clinical trial datasets with comprehensive validation
- **Metabolic Syndrome Identification**: Implements ATP III guidelines for patient identification
- **Advanced Feature Engineering**: Temporal analysis of lipid panels and blood pressure data
- **Multiple Risk Prediction Algorithms**: Framingham Risk Score, Pooled Cohort Equation, and custom models
- **Comprehensive Statistical Validation**: AUC-ROC, calibration analysis, subgroup validation
- **Regulatory Documentation**: FDA 510(k) submission artifacts and statistical analysis plans
- **Interactive Visualizations**: Clinical data exploration and validation result visualization
- **Configuration-Driven Architecture**: Flexible, parameterized pipeline configuration

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-org/cardiovascular-risk-prediction.git
cd cardiovascular-risk-prediction
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Initialize configuration**:
```bash
python -m src.cli init-config
```

### Basic Usage

1. **Ingest clinical data**:
```bash
python -m src.cli ingest -i data/clinical_data.csv -o outputs/ingestion
```

2. **Clean and preprocess data**:
```bash
python -m src.cli clean -i outputs/ingestion/ingested_data.parquet -o outputs/cleaning
```

3. **Identify metabolic syndrome patients**:
```bash
python -m src.cli identify-metabolic-syndrome -i outputs/cleaning/cleaned_data.parquet -o outputs/metabolic_syndrome
```

4. **Train risk prediction model**:
```bash
python -m src.cli train -i outputs/metabolic_syndrome/metabolic_syndrome_data.parquet -o outputs/models --algorithm framingham
```

5. **Validate model performance**:
```bash
python -m src.cli validate -i outputs/metabolic_syndrome/metabolic_syndrome_data.parquet -m outputs/models/framingham_model.pkl -o outputs/validation
```

6. **Generate reports**:
```bash
python -m src.cli report -i outputs/validation/validation_results.json -o outputs/reports
```

## 📊 Pipeline Workflow

```mermaid
graph TD
    A[Clinical Data] --> B[Data Ingestion]
    B --> C[Data Cleaning]
    C --> D[Metabolic Syndrome Identification]
    D --> E[Feature Engineering]
    E --> F[Model Training]
    F --> G[Statistical Validation]
    G --> H[Subgroup Analysis]
    H --> I[Temporal Validation]
    I --> J[Regulatory Documentation]
    J --> K[FDA 510(k) Submission]
    
    B --> L[EDA & Visualization]
    G --> M[Performance Metrics]
    H --> N[Equity Analysis]
    I --> O[Bias Assessment]
    
    style A fill:#e1f5fe
    style K fill:#c8e6c9
    style L fill:#fff3e0
    style M fill:#f3e5f5
    style N fill:#e8f5e8
    style O fill:#fff8e1
```

## 🏗️ Architecture

### Core Modules

#### 1. Data Processing (`src/cv_risk_pipeline/data/`)
- **`ingestion.py`**: Handles data loading from various sources (EHR, clinical trials)
- **`cleaning.py`**: Comprehensive data cleaning and quality assessment
- **`metabolic_syndrome.py`**: ATP III-based metabolic syndrome identification
- **`preprocessing.py`**: Feature engineering and data transformation

#### 2. Risk Prediction Models (`src/cv_risk_pipeline/models/`)
- **`risk_prediction.py`**: Abstract base class for risk prediction models
- **`algorithms.py`**: Implementations of established algorithms:
  - Framingham Risk Score
  - Pooled Cohort Equation
  - Custom machine learning models

#### 3. Statistical Validation (`src/cv_risk_pipeline/validation/`)
- **`statistical_validation.py`**: Performance metrics and calibration analysis
- **`subgroup_analysis.py`**: Equity analysis across patient subgroups
- **`temporal_validation.py`**: Performance stability and bias assessment

#### 4. Visualization (`src/cv_risk_pipeline/visualization/`)
- **`clinical_visualizer.py`**: Clinical data visualization
- **`eda_plots.py`**: Exploratory data analysis plots
- **`validation_plots.py`**: Model validation result visualization

#### 5. Regulatory Reporting (`src/cv_risk_pipeline/reporting/`)
- **`regulatory_reporter.py`**: Comprehensive validation reports
- **`fda_510k_reporter.py`**: FDA 510(k) submission documentation
- **`statistical_analysis_plan.py`**: Statistical analysis plan generation

#### 6. Utilities (`src/cv_risk_pipeline/utils/`)
- **`config_manager.py`**: Configuration management and validation
- **`logger.py`**: Logging setup and management

## 📋 Configuration

The pipeline uses YAML-based configuration files for all parameters. Key configuration sections:

### Data Configuration
```yaml
data:
  raw_data_path: "data/raw"
  processed_data_path: "data/processed"
  output_path: "outputs"
```

### Metabolic Syndrome Criteria
```yaml
metabolic_syndrome:
  criteria:
    waist_circumference:
      caucasian_male: 102
      caucasian_female: 88
    blood_pressure:
      systolic: 130
      diastolic: 85
    fasting_glucose: 100
    triglycerides: 150
    hdl_cholesterol:
      male: 40
      female: 50
  required_criteria: 3
```

### Model Configuration
```yaml
model:
  algorithm:
    name: "framingham_risk_score"
    parameters: {}
  cross_validation:
    n_folds: 5
    random_state: 42
```

### Validation Configuration
```yaml
validation:
  primary_metrics:
    - "auc_roc"
    - "sensitivity"
    - "specificity"
    - "ppv"
    - "npv"
  calibration:
    n_bins: 10
    method: "quantile"
```

## 🔬 Statistical Validation

### Primary Metrics
- **AUC-ROC**: Area under the receiver operating characteristic curve
- **Sensitivity**: True positive rate
- **Specificity**: True negative rate
- **PPV/NPV**: Positive and negative predictive values

### Secondary Metrics
- **Calibration Slope/Intercept**: Risk estimation accuracy
- **Brier Score**: Overall prediction accuracy
- **Hosmer-Lemeshow Test**: Goodness-of-fit assessment

### Subgroup Analysis
- Age groups (18-40, 40-60, 60-80, 80+ years)
- Sex-based analysis
- Baseline risk stratification
- Metabolic syndrome severity

### Temporal Validation
- Performance stability over time
- Seasonal pattern analysis
- Re-identification bias assessment

## 📈 Visualization Examples

### Clinical Data Exploration
- Longitudinal trajectory plots
- Risk factor distributions
- Correlation heatmaps
- Missing value analysis

### Model Validation
- ROC curves and precision-recall curves
- Calibration plots
- Confusion matrices
- Subgroup performance comparisons

### Interactive Dashboards
- Real-time performance monitoring
- Clinical decision support interfaces
- Regulatory compliance dashboards

## 📄 Regulatory Documentation

### FDA 510(k) Submission Package
- Device description and specifications
- Performance data and validation results
- Risk-benefit analysis
- Clinical evaluation report
- Software documentation
- Quality assurance procedures

### Statistical Analysis Plan
- Study design and objectives
- Sample size calculations
- Statistical methods and endpoints
- Subgroup analysis plans
- Safety monitoring procedures

### Validation Reports
- Executive summary
- Methodology and results
- Statistical analysis
- Subgroup and temporal validation
- Risk assessment and conclusions

## 🛠️ Advanced Usage

### Custom Risk Prediction Models

```python
from cv_risk_pipeline.models.risk_prediction import BlackBoxRiskModel

# Define custom prediction function
def custom_risk_function(X):
    # Your custom algorithm here
    return risk_scores

# Create black box model
config = {'model': {'algorithm': {'name': 'custom', 'parameters': {}}}}
model = BlackBoxRiskModel(config, custom_risk_function)

# Train and validate
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Configuration Management

```python
from cv_risk_pipeline.utils.config_manager import ConfigManager

# Load and modify configuration
config_manager = ConfigManager('config/default_config.yaml')
config_manager.update_config('model.algorithm.name', 'custom_model')
config_manager.save_config('config/custom_config.yaml')
```

### Batch Processing

```python
# Process multiple datasets
datasets = ['dataset1.csv', 'dataset2.csv', 'dataset3.csv']
for dataset in datasets:
    # Run complete pipeline for each dataset
    subprocess.run(['python', '-m', 'src.cli', 'run-pipeline', 
                   '-i', dataset, '-o', f'outputs/{dataset}'])
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/unit/
pytest tests/integration/

# Run with coverage
pytest --cov=src tests/
```

## 📚 Documentation

- **API Documentation**: [docs/api/](docs/api/)
- **User Guide**: [docs/user_guide.md](docs/user_guide.md)
- **Developer Guide**: [docs/developer_guide.md](docs/developer_guide.md)
- **Regulatory Guide**: [docs/regulatory_guide.md](docs/regulatory_guide.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run linting
black src/
flake8 src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏥 Clinical Disclaimer

This software is intended for research and development purposes. For clinical use, ensure compliance with applicable regulatory requirements and conduct appropriate validation studies.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-org/cardiovascular-risk-prediction/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/cardiovascular-risk-prediction/discussions)
- **Email**: support@clinical-ai.com

## 🙏 Acknowledgments

- Framingham Heart Study for risk prediction algorithms
- ATP III guidelines for metabolic syndrome criteria
- FDA guidance documents for regulatory compliance
- Open source community for foundational libraries

---

**Built with ❤️ for clinical decision support and patient care**


