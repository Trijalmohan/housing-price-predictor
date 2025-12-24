🏠 Housing Price per Sqft Prediction (Delhi NCR)

This project builds a machine learning pipeline to predict house price per square foot in the Delhi NCR region using structured real estate data.

📌 Features

Stratified train–test split

Custom feature engineering (City extraction from Address)

Robust preprocessing pipeline (imputation + scaling + encoding)

Models implemented:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Hyperparameter tuning using RandomizedSearchCV

Evaluation using RMSE

📊 Best Model Performance
Model	Test RMSE
Linear Regression	~2000
Decision Tree	~1500
Random Forest (tuned)	~1187 ✅
🧠 Pipeline Overview
Raw Data
   ↓
City Extraction (from Address)
   ↓
Numerical Pipeline (Imputer + Scaler)
   ↓
Categorical Pipeline (OneHotEncoder)
   ↓
Model (Random Forest)
   ↓
Prediction

🛠️ How to Run

1️⃣ Clone the repo:

git clone https://github.com/<your-username>/housing-price-predictor.git
cd housing-price-predictor


2️⃣ Install dependencies:

pip install -r requirements.txt


3️⃣ Add dataset:

data/Delhi_v2.csv


4️⃣ Run:

python model.py

🚀 Future Improvements

Extract locality/sector from Address

Log-transform target variable

Add text embeddings from property descriptions

Try XGBoost / LightGBM

⚠️ Note

Dataset is not included due to size/licensing constraints.