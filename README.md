# Housing Price Predictor 🏠

This project demonstrates how to build and compare multiple machine learning models for housing price prediction using real-world data.

The repository allows you to experiment with:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Hyperparameter tuning using GridSearchCV and RandomizedSearchCV
- Model evaluation using RMSE (Root Mean Squared Error)

Most model pipelines are already implemented, and alternative models or configurations are commented in the code so you can easily enable them and experiment further.

---

## 📊 Dataset

The dataset used in this project was obtained from **Kaggle**.

- Source: Kaggle – Delhi Housing Dataset  
- Target variable: `Price_sqft`

The dataset is **not included** in the repository.  
Download it from Kaggle and update the CSV path in the code before running.

---

## ⚙️ Features Used

Numerical features:
- Area
- Latitude
- Longitude
- Bedrooms
- Bathrooms
- Balcony
- Parking
- Lift

Categorical features:
- City (extracted from Address using a custom transformer)

---

## 🧠 Models & Techniques

- Custom preprocessing pipeline using `Pipeline` and `ColumnTransformer`
- Custom `CityExtractor` transformer
- Stratified train-test split
- Cross-validation using RMSE
- Hyperparameter tuning with:
  - GridSearchCV
  - RandomizedSearchCV

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/Trijalmohan/housing-price-predictor.git

2.Install dependencies:

pip install -r requirements.txt

3.Update dataset path in the code:

pd.read_csv("path/to/Delhi_v2.csv")

4.Run the model:

python model.py

📈 Results

After hyperparameter tuning using RandomizedSearchCV, the model achieved a significantly lower RMSE, demonstrating the effectiveness of ensemble methods like Random Forest for this problem.

🧪 Experimentation

Feel free to uncomment different model pipelines and tuning strategies in the code to:

Compare performance

Adjust hyperparameters

Try different feature combinations
This project is designed for learning, experimentation, and extension.