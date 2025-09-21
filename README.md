# PGNO - 151

## Overview of the Project

This project is all about creating a machine learning model to predict stroke risk using health and lifestyle data. Our focus for this Progress Review I is to clean and prep the dataset with preprocessing techniques and dig into it with Exploratory Data Analysis (EDA). The aim is to build a solid pipeline that gets the data ready for modeling, helping us ace the viva and boost our final evaluation marks.

## Dataset Details

We’re using `StrokeData.csv`, stored in the `data/raw/` folder. It’s the Kaggle Stroke Prediction Dataset with 5110 rows and 12 columns: id, gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status, and stroke (the target). We’ve noticed some challenges—about 4% missing bmi values, possible outliers in glucose and bmi, categorical variables that need encoding, and a big imbalance with only ~5% stroke cases. No extra datasets are in play yet.

## Group Member Roles

Here’s how we split the work for preprocessing. Each of us tackled a technique, created a notebook with code, explained why it’s needed, showed it working, and added an EDA viz with a clear takeaway.

*   **IT Number:** IT24102046 - Kuruppuarachchige K.A.H.B
    *   **Subtask:** Dimension Reduction (PCA)
    *   **Notebook:** `IT24102046_Dimension_Reduction_PCA.ipynb`
    *   **Scenario:** After encoding, we end up with around 20 features, which can make training slow and lead to overfitting.
    *   **Explanation:** I used PCA to cut it down to 8 components, keeping 95% of the variance. The scatter plot shows stroke clusters forming, which should help the model spot patterns better.

*   **IT Number:** IT24102070 - Pihara H.G.T
    *   **Subtask:** Feature Engineering - Selection
    *   **Notebook:** `IT24102070_Feature_Selection.ipynb`
    *   **Scenario:** Some features, like Residence_type, don’t seem to link much to stroke and just add noise.
    *   **Explanation:** I picked the top 8 features with SelectKBest and chi-squared. The correlation matrix highlights age and glucose as key players, and it shows age and ever_married are linked (0.68), so selection cleans things up.

*   **IT Number:** IT24102181 - Dilshan R.M.R
    *   **Subtask:** Handling Missing Data (bmi imputation)
    *   **Notebook:** `IT24102181_Handling_Missing_Data_bmi_imputation.ipynb`
    *   **Scenario:** We’ve got 201 missing bmi values (4%), and dropping them would shrink our already rare stroke cases.
    *   **Explanation:** I filled in the gaps with the median since bmi is a bit skewed. The histogram shows the distribution holds steady (mean ~28.9), so we avoid skewing the data for stroke prediction.

*   **IT Number:** IT24102008 - Withana W.Y.P
    *   **Subtask:** Encoding Categorical Variables
    *   **Notebook:** `IT24102008_Encoding_Categorical_Variables.ipynb`
    *   **Scenario:** String categories like work_type (5 options) and smoking_status (4) need to be numbers for the model to use them.
    *   **Explanation:** I used one-hot encoding for the multi-class ones and mapped binary ones (e.g., gender Male=0/Female=1, treating rare ‘Other’ as Female). The bar plot reveals former smokers have a higher stroke rate (~10%), which supports why we encoded this.

*   **IT Number:** IT24102131 - De Silva P.K.N
    *   **Subtask:** Outlier Removal
    *   **Notebook:** `IT24102131_Outlier_Removal.ipynb`
    *   **Scenario:** Glucose over 200 and bmi over 50 pop up as outliers, but they’re common in stroke data and could be important.
    *   **Explanation:** I capped them at the 99th percentile with quantiles to keep the data usable. The boxplot shows less spread afterward, which should stabilize the model without losing high-risk cases.

*   **IT Number:** IT24100618 - Inshaf M J M
    *   **Subtask:** Normalization/Scaling
    *   **Notebook:** `IT24100618_Normalization_Scaling.ipynb`
    *   **Scenario:** Numbers like age (0-82), glucose (55-271), and bmi (10-97) are on different scales, which can throw off algorithms.
    *   **Explanation:** I used StandardScaler to get everything to a mean of 0 and std of 1. The density plot by stroke shows risk jumps after age 60 (scaled >0.5), proving scaling levels the playing field for the model.

## How to Run the Code

We’ve made it straightforward to set up and run for the viva.

1.  **Prerequisites:** Python 3.8+, Jupyter Notebook. You’ll need pandas, numpy, scikit-learn, matplotlib, and seaborn.
2.  **Installation:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
3.  **Execution:**
    To run the individual notebooks, navigate to the `notebooks/` directory and open them with Jupyter.
    To run the integrated pipeline, execute `group_pipeline.ipynb` in the `notebooks/` directory.

## Repository Layout

```
PG151MLProject/
├── README.md
├── data/
│   ├── raw/             # Original dataset(s)
│   └── external/        # Any external reference datasets (if used)
├── notebooks/
│   ├── IT24102046_Dimension_Reduction_PCA.ipynb # Member 1 - IT24102046 & Dimension Reduction (PCA)
│   ├── IT24102070_Feature_Selection.ipynb # Member 2 - IT24102070 & Feature Engineering - Selection
│   ├── IT24102181_Handling_Missing_Data_bmi_imputation.ipynb # Member 3 - IT24102181 & Handling Missing Data (bmi imputation)
│   ├── IT24102008_Encoding_Categorical_Variables.ipynb # Member 4 - IT24102008 & Encoding Categorical Variables
│   ├── IT24102131_Outlier_Removal.ipynb # Member 5 - IT24102131 & Outlier Removal
│   ├── IT24100618_Normalization_Scaling.ipynb # Member 6 - IT24100618 & Normalization/Scaling
│   └── group_pipeline.ipynb   # Integrated pipeline (combined work)
└── results/
    ├── eda_visualizations/  # Plots & charts (PNG/JPEG)
    ├── logs/                # Any logs from execution (Optional)
    └── outputs/             # Final processed dataset / features
```