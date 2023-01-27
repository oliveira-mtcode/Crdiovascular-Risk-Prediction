"""
Statistical analysis plan module for cardiovascular risk prediction pipeline.

This module provides comprehensive statistical analysis planning capabilities
for clinical validation studies and regulatory submissions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from pathlib import Path
from datetime import datetime
import yaml
import json
from jinja2 import Template
import warnings

logger = logging.getLogger(__name__)


class StatisticalAnalysisPlan:
    """
    Provides statistical analysis planning capabilities for cardiovascular risk prediction models.
    
    Generates comprehensive statistical analysis plans including study design,
    sample size calculations, and analysis methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize statistical analysis plan with configuration.
        
        Args:
            config: Configuration dictionary containing analysis parameters
        """
        self.config = config
        self.analysis_config = config.get('validation', {})
        
        logger.info("Statistical analysis plan initialized")
    
    def generate_analysis_plan(self, output_path: str) -> str:
        """
        Generate comprehensive statistical analysis plan.
        
        Args:
            output_path: Path to save the analysis plan
            
        Returns:
            Path to the generated plan
        """
        logger.info("Generating statistical analysis plan")
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate plan sections
        plan_sections = {
            'study_design': self._generate_study_design_section(),
            'sample_size': self._generate_sample_size_section(),
            'statistical_methods': self._generate_statistical_methods_section(),
            'analysis_populations': self._generate_analysis_populations_section(),
            'primary_endpoints': self._generate_primary_endpoints_section(),
            'secondary_endpoints': self._generate_secondary_endpoints_section(),
            'subgroup_analyses': self._generate_subgroup_analyses_section(),
            'safety_analyses': self._generate_safety_analyses_section(),
            'interim_analyses': self._generate_interim_analyses_section(),
            'data_handling': self._generate_data_handling_section()
        }
        
        # Generate HTML plan
        html_plan = self._generate_html_plan(plan_sections)
        html_path = output_path / 'statistical_analysis_plan.html'
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_plan)
        
        # Generate JSON summary
        json_summary = self._generate_json_summary(plan_sections)
        json_path = output_path / 'analysis_plan_summary.json'
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_summary, f, indent=2, default=str)
        
        logger.info(f"Statistical analysis plan generated at {output_path}")
        return str(output_path)
    
    def _generate_study_design_section(self) -> str:
        """Generate study design section."""
        return """
        <h2>1. Study Design</h2>
        
        <h3>1.1 Study Type</h3>
        <p><strong>Retrospective Validation Study</strong></p>
        <p>This study employs a retrospective cohort design to validate the cardiovascular risk 
        prediction model using historical clinical data from electronic health records and 
        clinical trial databases.</p>
        
        <h3>1.2 Study Objectives</h3>
        <h4>Primary Objective</h4>
        <p>To validate the diagnostic performance of the cardiovascular risk prediction model 
        in patients with metabolic syndrome, as measured by the Area Under the ROC Curve (AUC-ROC).</p>
        
        <h4>Secondary Objectives</h4>
        <ul>
            <li>To evaluate the sensitivity and specificity of the model</li>
            <li>To assess the positive and negative predictive values</li>
            <li>To evaluate model calibration and risk estimation accuracy</li>
            <li>To assess performance across patient subgroups</li>
            <li>To evaluate temporal stability of model performance</li>
        </ul>
        
        <h3>1.3 Study Population</h3>
        <p><strong>Target Population:</strong> Adult patients (≥18 years) with metabolic syndrome</p>
        <p><strong>Metabolic Syndrome Definition:</strong> ATP III criteria requiring ≥3 of 5 components:</p>
        <ul>
            <li>Elevated waist circumference (≥102 cm men, ≥88 cm women)</li>
            <li>Elevated blood pressure (≥130/85 mmHg)</li>
            <li>Elevated fasting glucose (≥100 mg/dL)</li>
            <li>Elevated triglycerides (≥150 mg/dL)</li>
            <li>Reduced HDL cholesterol (<40 mg/dL men, <50 mg/dL women)</li>
        </ul>
        
        <h3>1.4 Study Duration</h3>
        <p><strong>Data Collection Period:</strong> [To be specified based on available data]</p>
        <p><strong>Follow-up Period:</strong> 10 years for cardiovascular outcomes</p>
        <p><strong>Analysis Period:</strong> [To be specified]</p>
        """
    
    def _generate_sample_size_section(self) -> str:
        """Generate sample size calculation section."""
        return """
        <h2>2. Sample Size Calculation</h2>
        
        <h3>2.1 Sample Size Rationale</h3>
        <p>The sample size calculation is based on the primary endpoint (AUC-ROC) and aims to 
        provide sufficient power to detect clinically meaningful differences in model performance.</p>
        
        <h3>2.2 Primary Endpoint Sample Size</h3>
        <p><strong>Primary Endpoint:</strong> Area Under the ROC Curve (AUC-ROC)</p>
        <p><strong>Null Hypothesis:</strong> AUC-ROC = 0.5 (no discriminative ability)</p>
        <p><strong>Alternative Hypothesis:</strong> AUC-ROC > 0.5 (discriminative ability)</p>
        
        <h4>Sample Size Parameters</h4>
        <ul>
            <li><strong>Type I Error (α):</strong> 0.05 (two-sided)</li>
            <li><strong>Type II Error (β):</strong> 0.20 (Power = 80%)</li>
            <li><strong>Expected AUC-ROC:</strong> 0.75</li>
            <li><strong>Minimum Clinically Meaningful AUC-ROC:</strong> 0.65</li>
            <li><strong>Expected Prevalence:</strong> 20%</li>
        </ul>
        
        <h4>Sample Size Calculation</h4>
        <p>Using the method of Hanley and McNeil (1982) for AUC-ROC sample size calculation:</p>
        <ul>
            <li><strong>Required Sample Size:</strong> 1,000 patients</li>
            <li><strong>Expected Events:</strong> 200 cardiovascular events</li>
            <li><strong>Expected Non-events:</strong> 800 patients</li>
        </ul>
        
        <h3>2.3 Subgroup Analysis Sample Size</h3>
        <p>For subgroup analyses, the following sample sizes are targeted:</p>
        <ul>
            <li><strong>Age Groups:</strong> ≥200 patients per group</li>
            <li><strong>Sex-based Analysis:</strong> ≥400 patients per sex</li>
            <li><strong>Baseline Risk Groups:</strong> ≥150 patients per group</li>
        </ul>
        
        <h3>2.4 Sample Size Adjustments</h3>
        <p>The following adjustments may be applied:</p>
        <ul>
            <li><strong>Missing Data:</strong> 10% inflation factor</li>
            <li><strong>Data Quality:</strong> 5% inflation factor</li>
            <li><strong>Total Sample Size:</strong> 1,200 patients</li>
        </ul>
        """
    
    def _generate_statistical_methods_section(self) -> str:
        """Generate statistical methods section."""
        return """
        <h2>3. Statistical Methods</h2>
        
        <h3>3.1 General Statistical Principles</h3>
        <ul>
            <li>All analyses will be performed using appropriate statistical software</li>
            <li>Two-sided tests will be used with α = 0.05 unless otherwise specified</li>
            <li>Confidence intervals will be calculated at the 95% level</li>
            <li>Missing data will be handled using appropriate methods</li>
            <li>All statistical tests will be documented with assumptions and limitations</li>
        </ul>
        
        <h3>3.2 Primary Endpoint Analysis</h3>
        <h4>3.2.1 AUC-ROC Analysis</h4>
        <ul>
            <li><strong>Method:</strong> Non-parametric method (DeLong et al.)</li>
            <li><strong>Confidence Interval:</strong> 95% CI using bootstrap method</li>
            <li><strong>Statistical Test:</strong> One-sample t-test against null hypothesis AUC = 0.5</li>
            <li><strong>Effect Size:</strong> Cohen's d for AUC-ROC</li>
        </ul>
        
        <h4>3.2.2 Cross-Validation</h4>
        <ul>
            <li><strong>Method:</strong> 5-fold stratified cross-validation</li>
            <li><strong>Stratification:</strong> By outcome status</li>
            <li><strong>Metrics:</strong> Mean AUC-ROC, standard deviation, 95% CI</li>
            <li><strong>Repetition:</strong> 10 iterations with different random seeds</li>
        </ul>
        
        <h3>3.3 Secondary Endpoint Analyses</h3>
        <h4>3.3.1 Sensitivity and Specificity</h4>
        <ul>
            <li><strong>Method:</strong> Confusion matrix analysis</li>
            <li><strong>Confidence Intervals:</strong> Wilson score method</li>
            <li><strong>Statistical Test:</strong> Chi-square test for independence</li>
        </ul>
        
        <h4>3.3.2 Predictive Values</h4>
        <ul>
            <li><strong>Method:</strong> Direct calculation from confusion matrix</li>
            <li><strong>Confidence Intervals:</strong> Clopper-Pearson exact method</li>
            <li><strong>Dependency:</strong> Prevalence-dependent metrics</li>
        </ul>
        
        <h3>3.4 Calibration Analysis</h3>
        <h4>3.4.1 Calibration Slope and Intercept</h4>
        <ul>
            <li><strong>Method:</strong> Linear regression of observed vs. predicted probabilities</li>
            <li><strong>Binning:</strong> 10 equal-sized bins</li>
            <li><strong>Statistical Test:</strong> t-test for slope = 1, intercept = 0</li>
        </ul>
        
        <h4>3.4.2 Hosmer-Lemeshow Test</h4>
        <ul>
            <li><strong>Method:</strong> Chi-square goodness-of-fit test</li>
            <li><strong>Binning:</strong> 10 equal-sized bins</li>
            <li><strong>Statistical Test:</strong> Chi-square test with 8 degrees of freedom</li>
        </ul>
        
        <h3>3.5 Subgroup Analyses</h3>
        <h4>3.5.1 Age Group Analysis</h4>
        <ul>
            <li><strong>Groups:</strong> 18-40, 40-60, 60-80, 80+ years</li>
            <li><strong>Method:</strong> Separate AUC-ROC calculation per group</li>
            <li><strong>Statistical Test:</strong> ANOVA for group differences</li>
        </ul>
        
        <h4>3.5.2 Sex-based Analysis</h4>
        <ul>
            <li><strong>Groups:</strong> Male, Female</li>
            <li><strong>Method:</strong> Separate performance metrics per sex</li>
            <li><strong>Statistical Test:</strong> Two-sample t-test for AUC-ROC differences</li>
        </ul>
        
        <h3>3.6 Temporal Analysis</h3>
        <h4>3.6.1 Performance Over Time</h4>
        <ul>
            <li><strong>Method:</strong> Rolling window analysis</li>
            <li><strong>Window Size:</strong> 6 months</li>
            <li><strong>Statistical Test:</strong> Linear regression for trend analysis</li>
        </ul>
        
        <h4>3.6.2 Seasonal Analysis</h4>
        <ul>
            <li><strong>Method:</strong> Seasonal decomposition</li>
            <li><strong>Groups:</strong> Spring, Summer, Fall, Winter</li>
            <li><strong>Statistical Test:</strong> ANOVA for seasonal differences</li>
        </ul>
        """
    
    def _generate_analysis_populations_section(self) -> str:
        """Generate analysis populations section."""
        return """
        <h2>4. Analysis Populations</h2>
        
        <h3>4.1 Full Analysis Set (FAS)</h3>
        <p><strong>Definition:</strong> All patients with metabolic syndrome who have sufficient 
        data for risk prediction and outcome assessment.</p>
        
        <h4>Inclusion Criteria</h4>
        <ul>
            <li>Age ≥18 years</li>
            <li>Diagnosis of metabolic syndrome (ATP III criteria)</li>
            <li>Complete baseline risk factor data</li>
            <li>Minimum 1 year of follow-up data</li>
            <li>Valid outcome data (cardiovascular events)</li>
        </ul>
        
        <h4>Exclusion Criteria</h4>
        <ul>
            <li>Missing critical risk factor data (>50% missing)</li>
            <li>Invalid or implausible clinical values</li>
            <li>Insufficient follow-up time</li>
            <li>Withdrawal of consent</li>
        </ul>
        
        <h3>4.2 Per-Protocol Set (PPS)</h3>
        <p><strong>Definition:</strong> Patients from the FAS who meet all protocol requirements 
        and have complete data for all primary and secondary endpoints.</p>
        
        <h4>Additional Criteria</h4>
        <ul>
            <li>Complete risk factor data (no missing values)</li>
            <li>Valid model predictions</li>
            <li>Complete outcome data</li>
            <li>No protocol violations</li>
        </ul>
        
        <h3>4.3 Safety Analysis Set (SAS)</h3>
        <p><strong>Definition:</strong> All patients who received risk predictions, regardless 
        of completeness of data or protocol adherence.</p>
        
        <h4>Purpose</h4>
        <ul>
            <li>Assessment of device-related adverse events</li>
            <li>Evaluation of system performance issues</li>
            <li>Identification of technical problems</li>
        </ul>
        
        <h3>4.4 Subgroup Analysis Sets</h3>
        <h4>4.4.1 Age-based Subgroups</h4>
        <ul>
            <li>Young adults (18-40 years)</li>
            <li>Middle-aged (40-60 years)</li>
            <li>Elderly (60-80 years)</li>
            <li>Very elderly (80+ years)</li>
        </ul>
        
        <h4>4.4.2 Sex-based Subgroups</h4>
        <ul>
            <li>Male patients</li>
            <li>Female patients</li>
        </ul>
        
        <h4>4.4.3 Baseline Risk Subgroups</h4>
        <ul>
            <li>Low baseline risk (<10%)</li>
            <li>Moderate baseline risk (10-20%)</li>
            <li>High baseline risk (>20%)</li>
        </ul>
        """
    
    def _generate_primary_endpoints_section(self) -> str:
        """Generate primary endpoints section."""
        return """
        <h2>5. Primary Endpoints</h2>
        
        <h3>5.1 Primary Endpoint</h3>
        <p><strong>Area Under the ROC Curve (AUC-ROC)</strong></p>
        
        <h4>Definition</h4>
        <p>The AUC-ROC measures the model's ability to discriminate between patients who will 
        and will not experience cardiovascular events within 10 years.</p>
        
        <h4>Calculation Method</h4>
        <ul>
            <li><strong>Method:</strong> Non-parametric method (DeLong et al.)</li>
            <li><strong>Software:</strong> R package pROC or Python scikit-learn</li>
            <li><strong>Confidence Interval:</strong> 95% CI using bootstrap method</li>
        </ul>
        
        <h4>Statistical Analysis</h4>
        <ul>
            <li><strong>Primary Analysis:</strong> One-sample t-test against null hypothesis AUC = 0.5</li>
            <li><strong>Alternative Hypothesis:</strong> AUC > 0.5</li>
            <li><strong>Significance Level:</strong> α = 0.05 (one-sided)</li>
            <li><strong>Effect Size:</strong> Cohen's d = (AUC - 0.5) / SD</li>
        </ul>
        
        <h4>Success Criteria</h4>
        <ul>
            <li><strong>Minimum AUC-ROC:</strong> 0.65</li>
            <li><strong>Target AUC-ROC:</strong> 0.75</li>
            <li><strong>Statistical Significance:</strong> p < 0.05</li>
            <li><strong>Lower 95% CI:</strong> > 0.60</li>
        </ul>
        
        <h3>5.2 Primary Endpoint Analysis Plan</h3>
        <h4>5.2.1 Analysis Population</h4>
        <p>Primary analysis will be performed on the Per-Protocol Set (PPS).</p>
        
        <h4>5.2.2 Analysis Method</h4>
        <ul>
            <li>Calculate AUC-ROC for the full dataset</li>
            <li>Perform 5-fold stratified cross-validation</li>
            <li>Calculate mean AUC-ROC and 95% confidence interval</li>
            <li>Perform statistical test against null hypothesis</li>
        </ul>
        
        <h4>5.2.3 Sensitivity Analyses</h4>
        <ul>
            <li>Analysis on Full Analysis Set (FAS)</li>
            <li>Analysis with different cross-validation folds (3-fold, 10-fold)</li>
            <li>Analysis with different random seeds</li>
            <li>Analysis excluding outliers</li>
        </ul>
        """
    
    def _generate_secondary_endpoints_section(self) -> str:
        """Generate secondary endpoints section."""
        return """
        <h2>6. Secondary Endpoints</h2>
        
        <h3>6.1 Diagnostic Performance Metrics</h3>
        
        <h4>6.1.1 Sensitivity (True Positive Rate)</h4>
        <ul>
            <li><strong>Definition:</strong> Proportion of true positives correctly identified</li>
            <li><strong>Formula:</strong> TP / (TP + FN)</li>
            <li><strong>Target:</strong> ≥0.80</li>
            <li><strong>Statistical Test:</strong> One-sample proportion test</li>
        </ul>
        
        <h4>6.1.2 Specificity (True Negative Rate)</h4>
        <ul>
            <li><strong>Definition:</strong> Proportion of true negatives correctly identified</li>
            <li><strong>Formula:</strong> TN / (TN + FP)</li>
            <li><strong>Target:</strong> ≥0.70</li>
            <li><strong>Statistical Test:</strong> One-sample proportion test</li>
        </ul>
        
        <h4>6.1.3 Positive Predictive Value (PPV)</h4>
        <ul>
            <li><strong>Definition:</strong> Proportion of positive predictions that are correct</li>
            <li><strong>Formula:</strong> TP / (TP + FP)</li>
            <li><strong>Dependency:</strong> Prevalence-dependent</li>
            <li><strong>Statistical Test:</strong> One-sample proportion test</li>
        </ul>
        
        <h4>6.1.4 Negative Predictive Value (NPV)</h4>
        <ul>
            <li><strong>Definition:</strong> Proportion of negative predictions that are correct</li>
            <li><strong>Formula:</strong> TN / (TN + FN)</li>
            <li><strong>Dependency:</strong> Prevalence-dependent</li>
            <li><strong>Statistical Test:</strong> One-sample proportion test</li>
        </ul>
        
        <h3>6.2 Calibration Metrics</h3>
        
        <h4>6.2.1 Calibration Slope</h4>
        <ul>
            <li><strong>Definition:</strong> Slope of regression line (observed vs. predicted)</li>
            <li><strong>Target:</strong> 1.0 (perfect calibration)</li>
            <li><strong>Acceptable Range:</strong> 0.8 - 1.2</li>
            <li><strong>Statistical Test:</strong> t-test against null hypothesis slope = 1</li>
        </ul>
        
        <h4>6.2.2 Calibration Intercept</h4>
        <ul>
            <li><strong>Definition:</strong> Intercept of regression line</li>
            <li><strong>Target:</strong> 0.0 (perfect calibration)</li>
            <li><strong>Acceptable Range:</strong> -0.1 to 0.1</li>
            <li><strong>Statistical Test:</strong> t-test against null hypothesis intercept = 0</li>
        </ul>
        
        <h4>6.2.3 Brier Score</h4>
        <ul>
            <li><strong>Definition:</strong> Mean squared difference between predicted and observed outcomes</li>
            <li><strong>Formula:</strong> Σ(predicted - observed)² / n</li>
            <li><strong>Target:</strong> <0.25</li>
            <li><strong>Statistical Test:</strong> One-sample t-test</li>
        </ul>
        
        <h3>6.3 Additional Performance Metrics</h3>
        
        <h4>6.3.1 Accuracy</h4>
        <ul>
            <li><strong>Definition:</strong> Proportion of correct predictions</li>
            <li><strong>Formula:</strong> (TP + TN) / (TP + TN + FP + FN)</li>
            <li><strong>Target:</strong> ≥0.75</li>
        </ul>
        
        <h4>6.3.2 F1 Score</h4>
        <ul>
            <li><strong>Definition:</strong> Harmonic mean of precision and recall</li>
            <li><strong>Formula:</strong> 2 × (Precision × Recall) / (Precision + Recall)</li>
            <li><strong>Target:</strong> ≥0.70</li>
        </ul>
        
        <h4>6.3.3 Precision-Recall AUC</h4>
        <ul>
            <li><strong>Definition:</strong> Area under precision-recall curve</li>
            <li><strong>Target:</strong> ≥0.60</li>
            <li><strong>Statistical Test:</strong> One-sample t-test against null hypothesis</li>
        </ul>
        """
    
    def _generate_subgroup_analyses_section(self) -> str:
        """Generate subgroup analyses section."""
        return """
        <h2>7. Subgroup Analyses</h2>
        
        <h3>7.1 Pre-specified Subgroups</h3>
        
        <h4>7.1.1 Age-based Subgroups</h4>
        <ul>
            <li><strong>Young Adults:</strong> 18-40 years</li>
            <li><strong>Middle-aged:</strong> 40-60 years</li>
            <li><strong>Elderly:</strong> 60-80 years</li>
            <li><strong>Very Elderly:</strong> 80+ years</li>
        </ul>
        
        <h4>7.1.2 Sex-based Subgroups</h4>
        <ul>
            <li><strong>Male Patients</strong></li>
            <li><strong>Female Patients</strong></li>
        </ul>
        
        <h4>7.1.3 Baseline Risk Subgroups</h4>
        <ul>
            <li><strong>Low Risk:</strong> <10% baseline risk</li>
            <li><strong>Moderate Risk:</strong> 10-20% baseline risk</li>
            <li><strong>High Risk:</strong> >20% baseline risk</li>
        </ul>
        
        <h4>7.1.4 Metabolic Syndrome Severity</h4>
        <ul>
            <li><strong>Mild:</strong> 3 criteria</li>
            <li><strong>Moderate:</strong> 4 criteria</li>
            <li><strong>Severe:</strong> 5 criteria</li>
        </ul>
        
        <h3>7.2 Subgroup Analysis Methods</h3>
        
        <h4>7.2.1 Performance Metrics by Subgroup</h4>
        <ul>
            <li>Calculate AUC-ROC for each subgroup</li>
            <li>Calculate sensitivity, specificity, PPV, NPV</li>
            <li>Calculate 95% confidence intervals</li>
            <li>Perform statistical tests for subgroup differences</li>
        </ul>
        
        <h4>7.2.2 Statistical Tests</h4>
        <ul>
            <li><strong>ANOVA:</strong> For continuous variables (AUC-ROC)</li>
            <li><strong>Chi-square Test:</strong> For categorical variables</li>
            <li><strong>Two-sample t-test:</strong> For binary comparisons</li>
            <li><strong>Multiple Comparison Correction:</strong> Bonferroni or FDR</li>
        </ul>
        
        <h3>7.3 Interaction Analyses</h3>
        
        <h4>7.3.1 Age × Sex Interaction</h4>
        <ul>
            <li>Test for interaction between age and sex</li>
            <li>Stratified analysis by age and sex combinations</li>
            <li>Statistical test: Two-way ANOVA</li>
        </ul>
        
        <h4>7.3.2 Risk × Severity Interaction</h4>
        <ul>
            <li>Test for interaction between baseline risk and metabolic syndrome severity</li>
            <li>Stratified analysis by risk and severity combinations</li>
            <li>Statistical test: Two-way ANOVA</li>
        </ul>
        
        <h3>7.4 Post-hoc Subgroup Analyses</h3>
        <p>Additional subgroup analyses may be performed based on:</p>
        <ul>
            <li>Clinical findings from primary analyses</li>
            <li>Regulatory feedback</li>
            <li>Scientific literature updates</li>
        </ul>
        <p><strong>Note:</strong> Post-hoc analyses will be clearly labeled as exploratory.</p>
        """
    
    def _generate_safety_analyses_section(self) -> str:
        """Generate safety analyses section."""
        return """
        <h2>8. Safety Analyses</h2>
        
        <h3>8.1 Safety Endpoints</h3>
        
        <h4>8.1.1 Device-Related Adverse Events</h4>
        <ul>
            <li><strong>Definition:</strong> Any adverse event related to device use</li>
            <li><strong>Assessment:</strong> Causal relationship to device</li>
            <li><strong>Severity:</strong> Mild, Moderate, Severe</li>
            <li><strong>Outcome:</strong> Resolved, Ongoing, Fatal</li>
        </ul>
        
        <h4>8.1.2 System Performance Issues</h4>
        <ul>
            <li><strong>Definition:</strong> Technical problems with the system</li>
            <li><strong>Types:</strong> Software errors, data processing issues, interface problems</li>
            <li><strong>Impact:</strong> Clinical impact assessment</li>
            <li><strong>Resolution:</strong> Time to resolution</li>
        </ul>
        
        <h3>8.2 Safety Analysis Methods</h3>
        
        <h4>8.2.1 Adverse Event Analysis</h4>
        <ul>
            <li>Count and rate of adverse events</li>
            <li>Severity distribution</li>
            <li>Relationship to device</li>
            <li>Time to event analysis</li>
        </ul>
        
        <h4>8.2.2 System Performance Analysis</h4>
        <ul>
            <li>Frequency of system issues</li>
            <li>Types of problems encountered</li>
            <li>Impact on clinical workflow</li>
            <li>Resolution time and methods</li>
        </ul>
        
        <h3>8.3 Safety Monitoring</h3>
        
        <h4>8.3.1 Real-time Monitoring</h4>
        <ul>
            <li>Continuous system performance monitoring</li>
            <li>Automated error detection and reporting</li>
            <li>User feedback collection</li>
            <li>Performance metrics tracking</li>
        </ul>
        
        <h4>8.3.2 Periodic Safety Review</h4>
        <ul>
            <li>Monthly safety data review</li>
            <li>Quarterly safety report generation</li>
            <li>Annual safety assessment</li>
            <li>Regulatory reporting as required</li>
        </ul>
        
        <h3>8.4 Risk-Benefit Assessment</h3>
        
        <h4>8.4.1 Risk Assessment</h4>
        <ul>
            <li>Identification of potential risks</li>
            <li>Risk severity and probability assessment</li>
            <li>Risk mitigation strategies</li>
            <li>Residual risk evaluation</li>
        </ul>
        
        <h4>8.4.2 Benefit Assessment</h4>
        <ul>
            <li>Clinical benefits of the system</li>
            <li>Healthcare system benefits</li>
            <li>Patient outcome improvements</li>
            <li>Cost-effectiveness considerations</li>
        </ul>
        """
    
    def _generate_interim_analyses_section(self) -> str:
        """Generate interim analyses section."""
        return """
        <h2>9. Interim Analyses</h2>
        
        <h3>9.1 Interim Analysis Plan</h3>
        <p><strong>No interim analyses are planned for this validation study.</strong></p>
        
        <h4>Rationale</h4>
        <ul>
            <li>Retrospective study design with complete data</li>
            <li>No patient safety concerns requiring early stopping</li>
            <li>Sufficient sample size for definitive analysis</li>
            <li>Regulatory requirements for complete validation</li>
        </ul>
        
        <h3>9.2 Data Monitoring</h3>
        
        <h4>9.2.1 Continuous Monitoring</h4>
        <ul>
            <li>Data quality monitoring</li>
            <li>System performance tracking</li>
            <li>User feedback collection</li>
            <li>Technical issue identification</li>
        </ul>
        
        <h4>9.2.2 Periodic Review</h4>
        <ul>
            <li>Monthly data quality review</li>
            <li>Quarterly performance assessment</li>
            <li>Annual validation review</li>
            <li>Regulatory compliance check</li>
        </ul>
        
        <h3>9.3 Early Stopping Criteria</h3>
        <p><strong>No early stopping criteria are defined for this study.</strong></p>
        
        <h4>Rationale</h4>
        <ul>
            <li>Retrospective validation study</li>
            <li>Complete dataset available</li>
            <li>No safety concerns</li>
            <li>Regulatory requirements for full validation</li>
        </ul>
        
        <h3>9.4 Protocol Amendments</h3>
        
        <h4>9.4.1 Amendment Process</h4>
        <ul>
            <li>Documentation of rationale for changes</li>
            <li>Regulatory approval if required</li>
            <li>Version control and tracking</li>
            <li>Impact assessment on study integrity</li>
        </ul>
        
        <h4>9.4.2 Amendment Categories</h4>
        <ul>
            <li><strong>Administrative:</strong> Minor changes with no impact on study integrity</li>
            <li><strong>Substantial:</strong> Changes affecting study design or analysis</li>
            <li><strong>Major:</strong> Changes requiring regulatory approval</li>
        </ul>
        """
    
    def _generate_data_handling_section(self) -> str:
        """Generate data handling section."""
        return """
        <h2>10. Data Handling and Quality Assurance</h2>
        
        <h3>10.1 Data Collection</h3>
        
        <h4>10.1.1 Data Sources</h4>
        <ul>
            <li>Electronic Health Records (EHR)</li>
            <li>Clinical Trial Databases</li>
            <li>Laboratory Information Systems</li>
            <li>Pharmacy Systems</li>
        </ul>
        
        <h4>10.1.2 Data Elements</h4>
        <ul>
            <li>Demographic information</li>
            <li>Clinical measurements</li>
            <li>Laboratory results</li>
            <li>Medication history</li>
            <li>Outcome data</li>
        </ul>
        
        <h3>10.2 Data Quality Assurance</h3>
        
        <h4>10.2.1 Data Validation</h4>
        <ul>
            <li>Range checks for clinical values</li>
            <li>Consistency checks across data sources</li>
            <li>Completeness assessment</li>
            <li>Outlier detection and handling</li>
        </ul>
        
        <h4>10.2.2 Data Cleaning</h4>
        <ul>
            <li>Missing data imputation</li>
            <li>Duplicate record removal</li>
            <li>Data standardization</li>
            <li>Quality control procedures</li>
        </ul>
        
        <h3>10.3 Missing Data Handling</h3>
        
        <h4>10.3.1 Missing Data Patterns</h4>
        <ul>
            <li>Missing completely at random (MCAR)</li>
            <li>Missing at random (MAR)</li>
            <li>Missing not at random (MNAR)</li>
        </ul>
        
        <h4>10.3.2 Imputation Methods</h4>
        <ul>
            <li><strong>Complete Case Analysis:</strong> For primary analysis</li>
            <li><strong>Multiple Imputation:</strong> For sensitivity analysis</li>
            <li><strong>Last Observation Carried Forward:</strong> For longitudinal data</li>
            <li><strong>Mean/Median Imputation:</strong> For continuous variables</li>
        </ul>
        
        <h3>10.4 Data Security and Privacy</h3>
        
        <h4>10.4.1 Data Protection</h4>
        <ul>
            <li>Encryption of data in transit and at rest</li>
            <li>Access controls and authentication</li>
            <li>Audit logging of data access</li>
            <li>Compliance with privacy regulations</li>
        </ul>
        
        <h4>10.4.2 Data Anonymization</h4>
        <ul>
            <li>Removal of direct identifiers</li>
            <li>Data de-identification procedures</li>
            <li>Risk of re-identification assessment</li>
            <li>Data sharing protocols</li>
        </ul>
        
        <h3>10.5 Statistical Software and Tools</h3>
        
        <h4>10.5.1 Primary Software</h4>
        <ul>
            <li><strong>R:</strong> Statistical analysis and visualization</li>
            <li><strong>Python:</strong> Machine learning and data processing</li>
            <li><strong>SAS:</strong> Regulatory reporting and validation</li>
        </ul>
        
        <h4>10.5.2 Version Control</h4>
        <ul>
            <li>Software version documentation</li>
            <li>Package version tracking</li>
            <li>Reproducibility procedures</li>
            <li>Code validation and testing</li>
        </ul>
        """
    
    def _generate_html_plan(self, plan_sections: Dict[str, str]) -> str:
        """Generate HTML plan from sections."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Statistical Analysis Plan - Cardiovascular Risk Prediction Model</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                h1 { color: #2E86AB; border-bottom: 2px solid #2E86AB; }
                h2 { color: #A23B72; margin-top: 30px; }
                h3 { color: #F18F01; }
                h4 { color: #C73E1D; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                ul { margin: 10px 0; }
                li { margin: 5px 0; }
                .header { text-align: center; margin-bottom: 40px; }
                .footer { margin-top: 40px; text-align: center; color: #666; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Statistical Analysis Plan</h1>
                <h2>Cardiovascular Risk Prediction Model Validation Study</h2>
                <p><strong>Version:</strong> 1.0</p>
                <p><strong>Date:</strong> {date}</p>
            </div>
            
            {study_design}
            {sample_size}
            {statistical_methods}
            {analysis_populations}
            {primary_endpoints}
            {secondary_endpoints}
            {subgroup_analyses}
            {safety_analyses}
            {interim_analyses}
            {data_handling}
            
            <div class="footer">
                <p>This statistical analysis plan was generated by the Cardiovascular Risk Prediction Pipeline</p>
                <p>For questions or clarifications, please contact the statistical team</p>
            </div>
        </body>
        </html>
        """
        
        template = Template(html_template)
        return template.render(
            date=datetime.now().strftime("%Y-%m-%d"),
            study_design=plan_sections['study_design'],
            sample_size=plan_sections['sample_size'],
            statistical_methods=plan_sections['statistical_methods'],
            analysis_populations=plan_sections['analysis_populations'],
            primary_endpoints=plan_sections['primary_endpoints'],
            secondary_endpoints=plan_sections['secondary_endpoints'],
            subgroup_analyses=plan_sections['subgroup_analyses'],
            safety_analyses=plan_sections['safety_analyses'],
            interim_analyses=plan_sections['interim_analyses'],
            data_handling=plan_sections['data_handling']
        )
    
    def _generate_json_summary(self, plan_sections: Dict[str, str]) -> Dict[str, Any]:
        """Generate JSON summary of analysis plan."""
        return {
            'plan_metadata': {
                'version': '1.0',
                'generated_at': datetime.now().isoformat(),
                'plan_type': 'statistical_analysis_plan',
                'study_type': 'retrospective_validation'
            },
            'study_design': {
                'study_type': 'retrospective_validation_study',
                'primary_objective': 'validate_diagnostic_performance',
                'target_population': 'patients_with_metabolic_syndrome',
                'follow_up_period': '10_years'
            },
            'sample_size': {
                'total_sample_size': 1200,
                'expected_events': 200,
                'expected_non_events': 800,
                'power': 0.80,
                'alpha': 0.05
            },
            'primary_endpoint': {
                'endpoint': 'auc_roc',
                'target_value': 0.75,
                'minimum_value': 0.65,
                'statistical_test': 'one_sample_t_test'
            },
            'secondary_endpoints': [
                'sensitivity',
                'specificity',
                'positive_predictive_value',
                'negative_predictive_value',
                'calibration_slope',
                'calibration_intercept',
                'brier_score'
            ],
            'subgroup_analyses': [
                'age_groups',
                'sex_based',
                'baseline_risk_groups',
                'metabolic_syndrome_severity'
            ],
            'statistical_methods': {
                'primary_analysis': 'per_protocol_set',
                'cross_validation': '5_fold_stratified',
                'missing_data_handling': 'complete_case_analysis',
                'multiple_comparisons': 'bonferroni_correction'
            }
        }
