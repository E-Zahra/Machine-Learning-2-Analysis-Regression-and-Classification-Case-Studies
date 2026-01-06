
Drinking Behavior Classification
======================================================

----------------
Project Overview
----------------
This classification task uses a dataset from Kaggle:
https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset

The dataset was collected from the National Health Insurance Service in Korea. All personal information and sensitive data were excluded.

Total entries: 991,346
Total columns: 24 (including target variable)

The objective of this project is to predict drinking behavior (Yes / No) using various machine learning classification models.



-------------------
Dataset Description
-------------------
Target Variable:
- DRK_YN: Drinker or Not (Yes / No)

Feature descriptions include demographics, vision, hearing, blood pressure, blood chemistry, liver enzymes, smoking status, and other health indicators.

Detailed description of the dataset can be found in Jupyter notebook, or in the PDF / HTML files.



-----------------
Project Structure
-----------------
code.ipynb:
Complete codes for EDA, feature selection, feature engineering, model training, and evaluation.

drinking_split.pkl:
Dataset used for EDA, feature selection, and engineering.

drinking_finalized.pkl:
Final dataset for model training.

catboost.pkl / decision_tree.pkl / random_forest.pkl / xgboost.pkl:
Saved randomized CV search results, runtime records, and best parameters, for model evaluation and final test.

To use the codes without training model, please run the pkl files directly. Detailed comments can be found in the code.ipynb.



----------------------
Data Preparation & EDA
----------------------
Dataset split before EDA to avoid data leakage:
- Train: 594,807 rows (60%)
- Validation: 198,269 rows (20%)
- Test: 198,270 rows (20%)

- Univariate analysis for target variable, categorical variables and numeric variables.
- Bivariate analysis with target feature for numeric and categorical variables were conducted one by one separately using boxplots, kernel density estimation plots and barplots separately, for inspecting the relationship between input variables and target variable carefully.

EDA findings:
- Balanced target distribution
- Many numeric outliers
- Most numeric features are non-normally distributed



-----------------
Feature Selection
-----------------
Methods used:
- Spearman correlation (for numeric only due to high amount of outliers)
- ANOVA for categorical target and continuous predictors
- Cramér’s V for categorical target and nominal variables
- Spearman correlation for ordinal variables

Five variables were dropped based on statistical significance (F-statistics, higher, better) and correlation strength.



-------------------
Feature Engineering
-------------------
Log transformation applied to skewed numeric features.
Same transformations applied consistently across train, validation, and test sets.



---------------------------
Model Training & Evaluation
---------------------------
Models used:
- Decision Tree (baseline)
- Random Forest
- XGBoost
- CatBoost

Neural networks were not considered, as they typically underperform tree-based ensembles on structured tabular data because of limited feature dimensionality and frequency of data, and require significantly greater tuning and computational resources without clear performance gains.

ROC-AUC used as the primary evaluation metric.


Decision Tree (Baseline)
------------------------
TRAIN ROC-AUC: 0.8140
VALID ROC-AUC: 0.8083

- CART-based decision tree classifier was used as the baseline model.
- To control overfitting, strong pre-pruning regularization was applied through constraints on maximum depth, minimum samples per split, and minimum samples per leaf.
- The optimal model, selected via 5-fold stratified cross-validation, used entropy as the splitting criterion with a maximum depth of 13, requiring at least 1000 samples to split a node and 500 samples per leaf.
- Resulted in stable generalization performance.
- Computational cost is low (1.5 mins).
- Visualizations of performance v.s. depths and tree plots.


XGBoost
-------
TRAIN ROC-AUC: 0.8272
VALID ROC-AUC: 0.8205

- Captured complex non-linear interactions through gradient-boosted decision trees.
- Run time is short (4.5 mins).
- Strongly regularized using constraints on tree depth, minimum child weight, split gain (gamma), feature subsampling, and L1/L2 penalties.
- Hyperparameter tuning moderated learning rates combined with split regularization yielded optimal performance, minimized train–validation gap and stable generalization.
- Compared with different parameters e.g. learning rates and gamma, generated slightly different results, but more or less similar.


CatBoost
--------
TRAIN ROC-AUC: 0.8313
VALID ROC-AUC: 0.8210

- Advanced gradient boosting model leveraging ordered boosting and stochastic sampling to reduce prediction shift and overfitting.
- The optimal configuration which are a moderate learning rate with deeper trees and strong regularization.
- Compared to XGBoost, CatBoost provided marginal but consistent performance improvements which means that the dataset benefits from its handling of feature interactions and regularization strategy, but the computation time is a lot longer.
- Compared with different parameters generated slightly wore results, but more or less similar.
- Long run time around 38 mins.


Random Forest
-------------
TRAIN ROC-AUC: 0.8181
VALID ROC-AUC: 0.8141

- A bagging-based ensemble to reduce variance relative to a single decision tree.
- Subsample used to reduce the run time (total required for 45 mins).
- Bootstrap sampling and feature subsampling used to decorrelate trees, and out-of-bag estimation provided an internal generalization check.
- Compared with different parameters generated slightly wore results, but more or less similar.



----------------
Model Comparison
----------------
Across all evaluated models, XGBoost consistently achieved near-top predictive performance while maintaining strong generalization. 

Although CatBoost achieved a slightly higher ROC-AUC, the improvement over XGBoost was marginal on the order of 0.0005 to 0.001. 

It indicates that both models operate near the performance ceiling for the given feature set. 

The train–validation curves (can be found of the jupyter notebook) show that XGBoost demonstrates a small and stable generalization gap across all metrics. 

It shows that XGBoost achieves an effective bias–variance trade-off and has improved upon Random Forest and Decision Tree while remaining as stable as CatBoost. 

Regarding the computational cost, in large-scale settings (600k training samples), the efficiency difference is practically significant. 

Thus, XGBoost offers a more favorable balance between accuracy, interpretability, and efficiency. 



------------------
Final Test Results
------------------
XGBoost selected as the final model. Misclassification confidence plot was produced. 

TRAIN ROC-AUC: 0.8272
TEST ROC-AUC: 0.8199

The final XGBoost model demonstrates strong generalization on the held-out test set with minimal performance degradation relative to training and validation results. 

Precision and recall are well balanced across both classes which means no systematic classification bias. 

Analysis of prediction confidence reveals that most misclassifications occur near the decision threshold. 

There are well-calibrated probability estimates and limited high-confidence errors. 

In short, these results confirm the robustness and reliability of the selected model.
