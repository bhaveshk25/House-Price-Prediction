# House Price Prediction

![Domain](https://img.shields.io/badge/domain-Machine%20Learning-blue)
![Task](https://img.shields.io/badge/task-Regression-green)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Frameworks](https://img.shields.io/badge/frameworks-Scikit--Learn%20%7C%20TensorFlow-orange)

An end-to-end Machine Learning project to analyze residential housing data and predict property sale prices. This repository provides two complementary modeling pipelines: an interpretable **Linear Regression baseline** and an advanced **TensorFlow Decision Forests (TF-DF)** tabular model.

---

## 📌 Repository Structure

```
├── data/
│   ├── Housing.csv               # 545 records, 13 features (Baseline dataset)
│   ├── train.csv                 # 1,460 records, 81 columns (Ames Housing train)
│   ├── test.csv                  # 1,459 records, 80 columns (Ames Housing test)
│   └── sample_submission.csv     # Competition submission format
├── Housing.ipynb                 # Baseline Linear Regression model pipeline
├── notebook.ipynb                # Advanced TensorFlow Decision Forests pipeline
├── requirements.txt              # Project dependencies
├── source.json                   # Kaggle kernel reference & metadata
└── README.md                     # Project documentation
```

---

## 📊 Modeling Approaches & Comparison

| Feature / Aspect | Baseline Model (`Housing.ipynb`) | Advanced Model (`notebook.ipynb`) |
| :--- | :--- | :--- |
| **Dataset** | Housing Case Study | Ames Housing (Kaggle Advanced Regression) |
| **Data Size** | 545 samples × 13 features | 1,460 train / 1,459 test × 81 features |
| **Algorithm** | Linear Regression (`scikit-learn`) | Random Forest Regressor (`tensorflow_decision_forests`) |
| **Preprocessing** | Manual Label Encoding & Train/Test Split | Natively handles numeric, categorical, and missing values |
| **Evaluation Metrics** | MSE, RMSE, $R^2$ Score | Out-of-Bag (OOB) RMSE, Validation MSE/RMSE, Variable Importance |
| **Use Case** | Quick baseline, interpretable feature coefficients | Production-grade tabular baseline, high non-linear capacity |

---

## 🛠️ Datasets

Both datasets are bundled in the `data/` folder for immediate out-of-the-box execution:

1. **Ames Housing Dataset (`data/train.csv`, `data/test.csv`)**:
   - 1,460 training examples with 79 explanatory variables + `Id` + `SalePrice`.
   - Contains mixed numerical features (e.g., `GrLivArea`, `LotArea`, `YearBuilt`) and categorical features (e.g., `Neighborhood`, `SaleCondition`).
   - Sourced from the [Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) competition.

2. **Housing Case Study Dataset (`data/Housing.csv`)**:
   - 545 residential properties with 12 features (`area`, `bedrooms`, `bathrooms`, `stories`, `parking`, `furnishingstatus`, etc.) and `price`.

---

## 🚀 Machine Learning Pipelines

### 1. Advanced Pipeline (`notebook.ipynb`)
- **Exploratory Data Analysis**: Inspects distribution of `SalePrice` using Seaborn histograms and reviews distributions across all 80 numerical and categorical features.
- **Dataset Splitting**: Splits data into 70% train and 30% validation sets with conversion to `tf.data.Dataset` via `tfdf.keras.pd_dataframe_to_tf_dataset`.
- **Model Training**: Trains a `RandomForestModel(task=REGRESSION)` using tree ensembles without requiring manual feature scaling or one-hot encoding.
- **Model Inspection**: Visualizes decision trees and inspects variable importance (`NUM_AS_ROOT`, `SUM_SCORE`).
- **Out-of-Bag (OOB) Evaluation**: Tracks out-of-bag RMSE as trees are built and calculates validation MSE.
- **Submission Generation**: Generates predictions for `test.csv` and outputs `submission.csv`.

### 2. Baseline Pipeline (`Housing.ipynb`)
- **Data Inspection & Cleaning**: Inspects missing values, summary statistics, and categorical distributions.
- **Feature Preprocessing**: Applies `LabelEncoder` to categorical attributes.
- **Training**: Fits a `LinearRegression` model using an 80/20 train/test split.
- **Evaluation**: Calculates Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and $R^2$ score.
- **Interactive Inference**: Includes a modular `predict_house_price()` function to estimate prices for custom property configurations.

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9+ installed
- Recommended: A virtual environment (`venv` or `conda`)

### Step-by-Step Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/bhaveshk25/House-Price-Prediction.git
   cd House-Price-Prediction
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
   - Open `Housing.ipynb` to run the baseline Linear Regression model.
   - Open `notebook.ipynb` to run the TensorFlow Decision Forests pipeline.

> **Note on TensorFlow Decision Forests**: TF-DF wheels are readily available on Linux and x86_64 macOS. If running on Apple Silicon (M1/M2/M3) or Windows, running `notebook.ipynb` in [Google Colab](https://colab.research.google.com/) is recommended.

---

## 📈 Results & Key Takeaways

- **No manual preprocessing required for TF-DF**: TensorFlow Decision Forests natively handle missing data, mixed numeric/categorical types, and non-linear interactions without feature imputation or scaling.
- **Validation Signal**: Out-of-Bag (OOB) error monitoring provides a reliable internal validation estimate as tree depth increases.
- **Extensibility**: The pipeline can easily be extended to `GradientBoostedTreesModel` or tuned via hyperparameter search for even higher predictive accuracy.

---

## 📚 Credits & Acknowledgements

- **TF-DF Architecture**: Inspired by the Kaggle notebook ["House Prices Prediction using TFDF"](https://www.kaggle.com/code/gusthema/house-prices-prediction-using-tfdf) by Gusthema.
- **Competition Dataset**: [Kaggle House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).
