# House Price Prediction: Advanced Regression Techniques

This project tackles the **House Prices - Advanced Regression Techniques** competition on Kaggle. The objective is to predict the selling prices of residential homes based on various property features such as area, location, number of rooms, construction year, etc. The solution leverages advanced regression techniques using LightGBM.

---

## About the Project

The goal of this project is to build a robust machine learning model that can accurately predict house prices based on a dataset of historical sales. The final predictions are evaluated using Root Mean Squared Logarithmic Error (RMSLE), emphasizing equal penalty on errors regardless of whether a house is expensive or affordable.

---

## Dataset Overview

The dataset comes from the Ames Housing dataset used in the Kaggle competition:

- **Train set:** 1460 entries with 81 features + target (`SalePrice`)
- **Test set:** 1459 entries (missing the target column)
- Features include numerical and categorical attributes related to the physical characteristics of the house, its location, and condition.

  **Dataset Source:** [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/)

---

## Approach

The solution involves:
- Comprehensive **data preprocessing** (handling missing values, outlier analysis, encoding)
- Extensive **feature engineering** to enhance the model's learning capacity
- Use of **ordinal encoding** for quality-related fields and **label encoding** for categorical features
- Log-transforming the target variable to better model price distribution
- Training a LightGBM Regressor with hyperparameter tuning and validation
- Generating final predictions on the test set and preparing a submission file

---

## Features Implemented

- Cleaned and well-structured dataset
- Missing value imputation (based on domain knowledge and statistics)
- Feature engineering:
  - Total living area
  - Total number of bathrooms
  - Age of the house
  - Presence of amenities like basement, garage, fireplace, and pool
- Encoding strategies:
  - Ordinal encoding for quality-related features
  - Label encoding for nominal categorical features
- Log transformation of the target (`SalePrice`) to normalize distribution
- Final model trained using LightGBM
- RMSE evaluation and submission file generated

---

## Tools and Technologies

- Python (Pandas, NumPy)
- Scikit-learn
- LightGBM
- Jupyter Notebook
- Matplotlib / Seaborn
- Git and GitHub

---

## How to Run the Project

```bash
# Clone the repository
git clone https://github.com/Prishatank0607/House-Price-Prediction.git

# Navigate to the project directory
cd House-Price-Prediction

# (Optional) Set up a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook
```
