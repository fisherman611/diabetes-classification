# Diabetes Classification

## Overview

This project focuses on building machine learning models to classify individuals as diabetic or non-diabetic using clinical data. The goal is to develop accurate predictive models that can support early diabetes detection and improve healthcare decision-making.

## Dataset

**Source**: [Diabetes Clinical Dataset](https://www.kaggle.com/datasets/ziya07/diabetes-clinical-dataset100k-rows/data) from Kaggle

**Size**: 100,000 records with 17 features

### Features

#### Demographic Information
- `year`: Year of record entry (2015-2022)
- `gender`: Patient's gender (Male, Female, Other)
- `age`: Age in years
- `location`: Geographical location
- `race`: One-hot encoded race categories (AfricanAmerican, Asian, Caucasian, Hispanic, Other)

#### Clinical Information
- `hypertension`: Presence of high blood pressure (1 = Yes, 0 = No)
- `heart_disease`: Presence of heart disease (1 = Yes, 0 = No)
- `smoking_history`: Smoking status (never, former, current, etc.)
- `bmi`: Body Mass Index
- `hbA1c_level`: Hemoglobin A1C level (average blood sugar over 2-3 months)
- `blood_glucose_level`: Blood glucose concentration
- `clinical_notes`: Additional clinical observations

#### Target Variable
- `diabetes`: Diabetes status (1 = Diabetic, 0 = Non-diabetic)

## Project Structure

```
diabetes-classification/
├── data/
│   ├── diabetes_dataset_with_notes.csv    # Original dataset
│   ├── feature_engineered_diabetes.csv    # After feature engineering
│   ├── preprocessed_diabetes.csv          # After preprocessing
│   ├── train_data.csv                     # Training set
│   └── test_data.csv                      # Test set
├── notebooks/
│   ├── EDA.ipynb                          # Exploratory Data Analysis
│   ├── feature_engineering.ipynb          # Feature engineering process
│   └── preprocess_data.ipynb              # Data preprocessing
├── models/
│   ├── logistic_regression.ipynb          # Logistic Regression model
│   ├── svm.ipynb                         # Support Vector Machine
│   ├── decision_tree.ipynb               # Decision Tree
│   ├── random_forest.ipynb               # Random Forest
│   ├── naive_bayes.ipynb                 # Naive Bayes
│   ├── xg_boost.ipynb                    # XGBoost
│   ├── light_gbm.ipynb                   # LightGBM
│   └── cat_boost.ipynb                   # CatBoost
├── utils/
│   ├── constants.py                       # Project constants
│   ├── function.py                        # Utility functions
│   └── split_dataset.py                  # Data splitting utilities
├── report.ipynb                           # Comprehensive project report
├── hba1c.jpg                             # HbA1c reference chart
└── README.md                             # This file
```

## Data Analysis Highlights

### Key Findings from EDA

1. **Dataset Balance**: The dataset is imbalanced with more non-diabetic cases
2. **Age Distribution**: Well-balanced across age groups, concentrated in 40-60 years
3. **Gender Distribution**: Female samples slightly higher than Male; "Other" category removed due to small representation
4. **BMI Categories**: 
   - Overweight: 43,519 individuals
   - Obese: 30,074 individuals
   - Normal: 24,869 individuals
   - Underweight: 1,531 individuals
5. **Clinical Indicators**:
   - HbA1c levels: 37,857 normal, 41,346 caution, 20,797 danger
   - Blood glucose: 51,372 normal, 37,751 prediabetic, 10,877 diabetic

## Data Preprocessing

### Steps Performed
1. **Data Cleaning**:
   - Removed 14 duplicate records
   - No missing values found
   
2. **Feature Engineering**:
   - Removed features: `year`, `smoking_history`, `clinical_notes`
   - Removed "Other" gender category
   - Combined low-frequency locations (Virgin Islands, Wisconsin, Wyoming) into "Others"
   - Added BMI classification (Underweight, Normal, Overweight, Obese)
   
3. **Data Transformation**:
   - Applied StandardScaler to numerical features: `age`, `bmi`, `hbA1c_level`, `blood_glucose_level`
   - Applied OneHotEncoder to categorical features: `gender`, `location`, `bmi_class`
   - Used stratified splitting to maintain class balance in train/test sets

## Machine Learning Models

This project implements 8 different machine learning algorithms to predict diabetes classification. Each model is carefully tuned and evaluated to find the optimal performance.

### 🤖 Models Implemented

#### 1. **Logistic Regression** (`logistic_regression.ipynb`)
- **Algorithm**: Linear classification with regularization
- **Configuration**: L1 regularization for feature selection
- **Strengths**: Interpretable coefficients, fast training
- **Use Case**: Baseline model for comparison and feature importance analysis

#### 2. **Support Vector Machine** (`svm.ipynb`)
- **Algorithm**: SVM with RBF (Radial Basis Function) kernel
- **Configuration**: Optimized C and gamma parameters
- **Strengths**: Effective in high-dimensional spaces, memory efficient
- **Use Case**: Complex decision boundaries with margin maximization

#### 3. **Decision Tree** (`decision_tree.ipynb`)
- **Algorithm**: CART (Classification and Regression Trees)
- **Configuration**: Optimized max_depth, min_samples_split, min_samples_leaf
- **Strengths**: Highly interpretable, handles non-linear relationships
- **Use Case**: Feature importance analysis and rule extraction

#### 4. **Random Forest** (`random_forest.ipynb`)
- **Algorithm**: Ensemble of decision trees with bagging
- **Configuration**: 100 estimators with optimized parameters
- **Strengths**: Reduces overfitting, handles missing values, robust
- **Use Case**: General-purpose ensemble with good performance

#### 5. **Naive Bayes** (`naive_bayes.ipynb`)
- **Algorithm**: Gaussian Naive Bayes classifier
- **Configuration**: Assumes feature independence
- **Strengths**: Fast training/prediction, works with small datasets
- **Use Case**: Probabilistic classification with assumption of feature independence

#### 6. **XGBoost** (`xg_boost.ipynb`)
- **Algorithm**: Extreme Gradient Boosting
- **Configuration**: Optimized learning rate, max_depth, n_estimators
- **Strengths**: State-of-the-art performance, handles missing values
- **Use Case**: High-performance competitive machine learning

#### 7. **LightGBM** (`light_gbm.ipynb`)
- **Algorithm**: Light Gradient Boosting Machine
- **Configuration**: Optimized num_leaves, learning_rate, feature_fraction
- **Strengths**: Fast training speed, low memory usage, high accuracy
- **Use Case**: Large datasets requiring efficient gradient boosting

#### 8. **CatBoost** (`cat_boost.ipynb`)
- **Algorithm**: Categorical Boosting
- **Configuration**: Optimized iterations, depth, learning_rate
- **Strengths**: Handles categorical features automatically, robust to overfitting
- **Use Case**: Tabular data with mixed feature types

### 🔧 Hyperparameter Tuning

Each model undergoes extensive hyperparameter optimization:

- **Method**: GridSearchCV with 5-fold cross-validation
- **Advanced Tuning**: Optuna framework for complex models (CatBoost, XGBoost, LightGBM)
- **Metrics**: Optimized for accuracy while balancing precision and recall
- **Validation**: Stratified sampling to maintain class distribution

### 📁 Model Checkpoints

All trained models are saved in the `checkpoints/` directory:
```
checkpoints/
├── catboost.pkl              # CatBoost model
├── decision_tree.pkl         # Decision Tree model
├── lgbm_model.pkl           # LightGBM model
├── logistic_regression.pkl   # Logistic Regression model
├── random_forest.pkl        # Random Forest model
├── svm.pkl                  # Support Vector Machine model
└── xgb_model.pkl            # XGBoost model
```

## Results

### Model Performance Comparison

| Model               | Accuracy | Recall  | Precision | F1 Score |
|---------------------|----------|---------|-----------|----------|
| Logistic Regression | 0.9609   | 0.6188  | 0.8870    | 0.7290   |
| SVM                 | 0.9690   | 0.6547  | 0.9712    | 0.7822   |
| Decision Tree       | 0.9722   | 0.6735  | **1.0000** | 0.8049   |
| Random Forest       | 0.9725   | 0.6829  | 0.9906    | 0.8085   |
| Naive Bayes         | 0.9563   | 0.5794  | 0.8618    | 0.6929   |
| XGBoost             | **0.9729**   | 0.6888  | 0.9899    | 0.8124   |
| LightGBM            | 0.9726   | 0.6877  | 0.9865    | 0.8104   |
| **CatBoost**        | **0.9729** | **0.6947** | 0.9809 | **0.8134** |

### Key Insights

1. **Top Performing Models**: **CatBoost and XGBoost** tie for highest accuracy (97.29%), with CatBoost achieving the best F1-score (0.8134)

2. **Performance Tiers**:
   - **Tier 1 (Elite)**: CatBoost (F1: 0.8134), XGBoost (F1: 0.8124) - State-of-the-art gradient boosting
   - **Tier 2 (Strong)**: LightGBM (F1: 0.8104), Random Forest (F1: 0.8085) - Excellent ensemble methods  
   - **Tier 3 (Solid)**: Decision Tree (F1: 0.8049), SVM (F1: 0.7822) - Good specialized performance
   - **Tier 4 (Baseline)**: Logistic Regression (F1: 0.7290), Naive Bayes (F1: 0.6929) - Traditional methods

3. **Precision vs Recall Trade-offs**:
   - **Highest Precision**: Decision Tree (100%) - Zero false positives but misses 32.65% of diabetic cases
   - **Best Recall**: CatBoost (69.47%) - Catches most diabetic cases while maintaining high precision (98.09%)
   - **Balanced Performance**: XGBoost and LightGBM show excellent precision-recall balance

4. **Feature Importance**: Top contributing features across models are:
   - **HbA1c level** (primary diabetes indicator)
   - **Blood glucose level** (critical biomarker)
   - **BMI** (obesity correlation)
   - **Age** (demographic risk factor)
   - **Heart disease** (comorbidity)
   - **Hypertension** (related condition)

5. **Clinical Implications**:
   - **For Screening**: Decision Tree's perfect precision minimizes unnecessary worry from false positives
   - **For Diagnosis**: CatBoost's balanced metrics make it ideal for comprehensive diabetes prediction
   - **For Research**: Gradient boosting models (CatBoost, XGBoost, LightGBM) consistently outperform traditional ML methods

## Requirements

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
catboost
optuna
jupyter
joblib
```

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/fisherman611/diabetes-classification.git
   cd diabetes-classification
   ```

2. **Set up environment**:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the analysis**:
   - Start with `notebooks/EDA.ipynb` for data exploration
   - Follow with preprocessing and feature engineering notebooks
   - Explore individual model notebooks in the `models/` directory
   - Review the comprehensive analysis in `report.ipynb`

## Future Improvements

### Identified Limitations
- **Low Recall**: Models miss approximately 30-40% of diabetic cases
- **Class Imbalance**: Dataset skewed toward non-diabetic cases

### Proposed Solutions
1. **Data Balancing**: Implement SMOTE, oversampling, or class weighting techniques
2. **Advanced Models**: ✅ **Implemented**: XGBoost, LightGBM, CatBoost | **Future**: Neural networks, deep learning approaches
3. **Feature Selection**: Remove low-importance features identified during analysis
4. **Ensemble Methods**: Combine multiple models for improved performance (model stacking/voting)
5. **Cost-sensitive Learning**: Adjust for the different costs of false positives vs false negatives
6. **Model Optimization**: Further hyperparameter tuning with advanced techniques like Bayesian optimization

## Clinical Significance

This project demonstrates how machine learning can support healthcare professionals in:
- **Early Detection**: Identifying at-risk individuals before symptoms appear
- **Resource Allocation**: Prioritizing patients for further testing
- **Preventive Care**: Enabling timely interventions to prevent complications

The high precision achieved by these models makes them suitable for screening applications where minimizing false positives is crucial.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for suggestions and improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset provided by [Kaggle](https://www.kaggle.com/datasets/ziya07/diabetes-clinical-dataset100k-rows/data)
- Healthcare domain expertise and clinical guidelines for diabetes diagnosis
