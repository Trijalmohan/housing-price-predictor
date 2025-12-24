# 🏠 Housing Price Predictor (Delhi NCR)

This repository contains a complete machine learning pipeline for predicting **housing prices per square foot** in the Delhi NCR region.

The goal of this project is to:
- Build and compare multiple regression models
- Tune model hyperparameters
- Evaluate model performance using RMSE
- Provide a reusable and extensible ML codebase

---

## 📊 Dataset

The dataset used in this project was obtained from **Kaggle**:

👉 https://www.kaggle.com/datasets/goelyash/housing-price-dataset-of-delhiindia

You must manually download the dataset from Kaggle and place it in the `data/` folder (this repo does *not* include the data due to size/privacy considerations).

---

## 🛠 Project Structure

housing-price-predictor/
│
├── data/
│ └── Delhi_v2.csv # Dataset (not included, download from Kaggle)
│
├── model.py # Main Python script
├── requirements.txt # Dependencies
├── README.md # Project documentation
├── .gitignore # Git ignore rules

yaml

---

## 🧠 What This Project Does

It demonstrates how to:

- Extract meaningful features (e.g., city from address)
- Build preprocessing pipelines using `ColumnTransformer` and `Pipeline`
- Compare models such as:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
- Apply hyperparameter tuning:
  - Grid Search
  - Randomized Search
- Evaluate model performance using **Root Mean Squared Error (RMSE)**

Commented sections of the code allow you to easily enable other models or experiments.

---

## ⚙️ How to Run

1. Clone the repo:
```bash
git clone https://github.com/Trijalmohan/housing-price-predictor.git
Install dependencies:

bash
pip install -r requirements.txt
Download the dataset from Kaggle and put it here:

kotlin

data/Delhi_v2.csv
Then update the path in the script if necessary:

python
data = pd.read_csv("data/Delhi_v2.csv")
Run the model script:

bash

python model.py
📈 Model Performance
Model	Test RMSE
Linear Regression	~2000
Decision Tree Regressor	~1500
Random Forest Regressor (tuned)	~1187

(values approximate and depend on dataset version and splits)

🧪 Experiment and Learn
Uncomment alternative model blocks in model.py to try them

Adjust hyperparameters or RandomizedSearch ranges

Add more features such as sectors or locality clustering

Try log-transform on the target variable

This repository is meant for experimentation and learning.

🚀 Next Steps
You could:

Add feature importance visualization

Build a web UI (Streamlit / Flask / FastAPI)

Deploy as an API

Add automated evaluation reports

📌 License
This work is open to anyone for learning and experimentation.
Please give credit if reused.









