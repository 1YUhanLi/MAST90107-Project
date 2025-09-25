# MAST90107 - Comprehensive Mammary Gland Cell Analysis

## Project Overview

This repository contains a comprehensive statistical and machine learning analysis of mammary gland cell behavior using TrackMate-detected microscopy data. The project addresses two fundamental research questions through rigorous analytical approaches: quantifying biological activity differences across cell models and developing automated cell type classification systems.

## Research Problems

### Problem A: Statistical Modeling of Spot Detection Rates
**Objective:** Quantify model-specific incidence of TrackMate-detected spots per frame across three biological models (ELF5, K5, PR) while controlling for imaging channel effects and exposure differences.

**Key Features:**
- Negative binomial regression with cluster-robust standard errors
- File-level aggregation to avoid pseudo-replication
- Exposure offset modeling using log(FRAMECOUNT)
- Comprehensive robustness validation and sensitivity analysis

### Problem B: Intelligent Cell Type Classification
**Objective:** Develop machine learning models for automated classification of mammary gland cells into three types (K5, ELF5, PR) based on morphological features.

**Key Challenges:**
- Severe class imbalance (K5: 253 samples, ELF5: 4,359 samples, PR: 9,172 samples)
- Feature engineering from microscopy measurements
- Model interpretability for biological insights

**Research Components:**
- **Primary Study**: Oversampling strategies (SMOTE, class weighting) for class imbalance
- **Comparative Study**: Downsampling approach vs. oversampling methods evaluation

## Project Structure

```
MAST90107/
├── data/
│   ├── raw/                     # 347 original TrackMate CSV files
│   └── processed/               # Analysis-ready datasets
│       ├── data_core.csv        # Main processed dataset
│       ├── ProblemA/            # Statistical analysis outputs
│       │   ├── A1_IRR_NegBin.csv
│       │   ├── A2_Adjusted_rate_by_MODEL.csv
│       │   └── ...
│       ├── ProblemB/            # ML analysis outputs
│       │   ├── data_enriched.csv
│       │   ├── primary_single_YFP_fold_scores.csv
│       │   └── ...
│       └── ProblemB_downsampling/  # Downsampling study results
│           ├── downsampling_fold_scores.csv
│           ├── method_comparison.csv
│           └── balanced_dataset_downsampled.csv
├── notebooks/
│   ├── eda.ipynb                # Exploratory Data Analysis
│   ├── feature_engineering.ipynb
│   ├── ProblemA_model_training_evaluation.ipynb
│   ├── ProblemB_model_traning_evaluation.ipynb
│   └── ProblemB_down_sampling.ipynb  # Downsampling vs oversampling study
├── Plots/                       # Visualization outputs
├── models/                      # Trained model artifacts
└── README.md
```

## Data Pipeline

### 1. Raw Data Processing
- **Input**: 347 TrackMate CSV files containing spot detection results
- **Processing**: Standardized schema merging with SOURCE_FILE tracking
- **Output**: `data_core.csv` with unified structure

### 2. Feature Engineering
- Morphological features (area, perimeter, circularity, aspect ratio)
- Intensity statistics (mean, median, standard deviation)
- Spatial features (centroid coordinates, bounding box dimensions)
- Temporal metadata (date extraction, batch effects)

### 3. Problem-Specific Data Preparation
- **Problem A**: File-level aggregation with exposure adjustments
- **Problem B**: Sample-level features with class balance considerations

## Problem A: Statistical Analysis

### Methodology
- **Model**: Negative Binomial GLM with exposure offset
- **Reference Design**: K5 (baseline model), RFP (reference channel)
- **Clustering**: Robust standard errors by date/file groups
- **Validation**: Poisson vs NegBin comparison, NB2 cross-check, sensitivity analysis

### Key Results
- **Model Effects**: ELF5 shows 3.66× higher incidence than K5 (95% CI: 2.78–4.81)
- **Channel Effects**: RFP provides optimal detection; CFP and YFP show 40% and 27% efficiency reduction
- **Standardized Rates**: ELF5 = 13.98, K5 = 3.82, PR = 10.94 spots/frame at RFP

### Generated Outputs
- Incidence rate ratios (IRR) with confidence intervals
- Channel-standardized detection rates
- Model diagnostic tables and plots
- Robustness validation reports

## Problem B: Machine Learning Classification

### Methodology

#### Primary Study: Oversampling Approach
- **Algorithms**: Random Forest, SVM, Logistic Regression, XGBoost
- **Class Balance**: SMOTE, ADASYN, cost-sensitive learning
- **Evaluation**: Stratified cross-validation, precision/recall/F1 metrics
- **Feature Selection**: Recursive feature elimination, importance ranking

#### Comparative Study: Downsampling vs. Oversampling
- **Objective**: Evaluate supervisor-recommended downsampling strategy against oversampling methods
- **Downsampling Strategy**: Random sampling of majority classes (ELF5, PR) to match minority class size (253 samples each)
- **Comparison Metrics**: Performance improvement, computational efficiency, class-specific F1-scores
- **Key Innovation**: Perfect class balance (1:1:1 ratio) with 94.5% data reduction

### Key Features
- **Morphological**: Shape descriptors, size measurements
- **Intensity**: Statistical summaries across channels
- **Spatial**: Position and distribution features
- **Quality**: Focus and acquisition metrics

### Generated Outputs

#### Primary Study (Oversampling)
- Model performance comparison tables
- Feature importance rankings
- Class-specific prediction metrics
- Balanced dataset experiments

#### Comparative Study (Downsampling vs. Oversampling)
- Method comparison analysis with statistical significance
- Computational efficiency benchmarks (18x training speedup)
- Class-wise F1-score improvements (K5: 0.667, ELF5: 0.592, PR: 0.892)
- Visualizations: confusion matrices, performance comparisons, efficiency plots

## Technical Implementation

### Statistical Analysis (Problem A)
```python
# Core analytical approach
negbin_model = sm.GLM(
    counts, design_matrix, 
    family=sm.families.NegativeBinomial(alpha=1.0),
    offset=np.log(frame_counts)
).fit(cov_type="cluster", cov_kwds={"groups": cluster_groups})
```

### Machine Learning (Problem B)

#### Oversampling Approach
```python
# Class-balanced classification pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('balancer', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(class_weight='balanced'))
])
```

#### Downsampling Approach
```python
# Balanced downsampling strategy
def apply_downsampling(dataframe, target_size=253, random_state=42):
    k5_data = dataframe[dataframe["META_MODEL"] == "K5"].copy()  # Keep all
    elf5_sampled = dataframe[dataframe["META_MODEL"] == "ELF5"].sample(n=target_size, random_state=random_state)
    pr_sampled = dataframe[dataframe["META_MODEL"] == "PR"].sample(n=target_size, random_state=random_state)
    return pd.concat([k5_data, elf5_sampled, pr_sampled], ignore_index=True)
```

## Key Dependencies

### Statistical Analysis
- `statsmodels`: GLM regression and diagnostics
- `patsy`: Design matrix construction
- `scipy`: Statistical tests and distributions

### Machine Learning
- `scikit-learn`: Classification algorithms and metrics
- `imbalanced-learn`: Class balancing techniques
- `xgboost`: Gradient boosting implementation

### Data Processing & Visualization
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `matplotlib/seaborn`: Statistical visualizations
- `pathlib`: File system operations

## Reproducibility

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Verify data structure
python -c "import pandas as pd; print(pd.read_csv('data/processed/data_core.csv').shape)"
```

### Analysis Execution
1. **Data Preparation**: Run `eda.ipynb` and `feature_engineering.ipynb`
2. **Statistical Analysis**: Execute `ProblemA_model_training_evaluation.ipynb`
3. **ML Classification (Primary)**: Execute `ProblemB_model_traning_evaluation.ipynb`
4. **ML Classification (Comparative)**: Execute `ProblemB_down_sampling.ipynb`

## Results Summary

### Problem A: Biological Activity Quantification
- **Statistical Validation**: Negative binomial regression confirmed over Poisson (AIC: 3,805 vs 34,731)
- **Biological Findings**: Clear hierarchy of activity (ELF5 > PR >> K5) with robust statistical support
- **Technical Insights**: RFP channel optimization confirmed across all models
- **Practical Impact**: Standardized benchmarks for experimental design and power calculations

### Problem B: Automated Classification

#### Primary Study Results (Oversampling)
- **Model Performance**: Achieved robust classification despite severe class imbalance
- **Feature Insights**: Morphological features provide strongest discriminative power
- **Methodological Contribution**: Effective strategies for handling minority class prediction
- **Biological Relevance**: Automated pipeline reduces manual classification workload

#### Comparative Study Results (Downsampling vs. Oversampling)
- **Performance Comparison**: Downsampling achieved +5.2% average improvement over oversampling
- **Best Model**: Logistic Regression with 0.519 Macro-F1 (vs. 0.457 with oversampling)
- **Computational Efficiency**: 94.5% data reduction with 18x training speedup
- **Class Balance Achievement**: Perfect 1:1:1 ratio while preserving all minority class samples

## Applications and Impact

### Research Applications
1. **Experimental Design**: Channel selection and sample size planning
2. **Cross-Study Comparisons**: Standardized incidence rate benchmarks
3. **Automated Analysis**: Scalable cell classification for large datasets
4. **Quality Control**: Systematic detection efficiency assessment

### Methodological Contributions
1. **Statistical Framework**: Robust approach for count data with technical confounders
2. **Class Imbalance Solutions**: Comprehensive comparison of oversampling vs. downsampling strategies
3. **Computational Efficiency**: Demonstrated 18x speedup with minimal performance trade-offs
4. **Validation Protocols**: Comprehensive robustness checking procedures
5. **Reproducible Pipeline**: End-to-end automated analysis workflow
6. **Practical Guidance**: Evidence-based recommendations for resource-constrained environments

## Citation and Acknowledgments

This project was developed as part of MAST90107 coursework, implementing state-of-the-art statistical and machine learning methods for biological image analysis. The analytical framework provides a foundation for systematic mammary gland cell research and automated microscopy analysis.

## Contact and Support

For questions regarding implementation details, analytical methods, or result interpretation, please refer to the detailed documentation within each notebook or contact the project maintainer.

---

