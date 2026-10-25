# 5G Traffic Prediction Using Machine Learning

A machine learning-based project for predicting 5G network bandwidth requirements using network measurements and an interactive Streamlit web application.

## Project Overview

The objective of this project is to analyze 5G network traffic data and predict the required bandwidth for a future observation. The application provides an interactive dashboard to explore network traffic, generate predictions, and examine model performance.

## Features

* Interactive Streamlit dashboard
* Network traffic data exploration and visualization
* Bandwidth prediction using a trained machine learning model
* Actual-versus-predicted comparison
* Model performance evaluation
* Model inference using a saved Joblib pipeline

## Technologies Used

* Python
* Pandas and NumPy
* Scikit-learn
* Joblib
* Streamlit
* Machine Learning — Regression

## Project Structure

```text
5G-Traffic-Prediction-ML/
├── app.py
├── 5G\_dataset.csv
├── 5G\_Traffic\_Final\_Lightweight\_Model.joblib
├── requirements.txt
├── README.md
├── run\_app.bat
├── .gitignore
└── screenshots/
```

## Model Performance

The existing project documentation reports the following test-set result:

* **Best model:** Extra Trees Regressor
* **R² score:** 0.8366

An R² score of 0.8366 indicates that the model explains approximately 83.66% of the variation in the target variable on the evaluated test set. R² is a regression metric, not classification accuracy.

Verify these results against your actual evaluation output before presenting them as final results.

## Installation and Setup

### Prerequisites

* Python version compatible with the project's dependencies
* pip

### 1\. Clone the repository

Replace `YOUR-USERNAME` with your GitHub username after creating the repository.

```bash
git clone https://github.com/YOUR-USERNAME/5G-Traffic-Prediction-ML.git
cd 5G-Traffic-Prediction-ML
```

### 2\. Create a virtual environment

```bash
python -m venv .venv
```

### 3\. Activate the environment on Windows

```bash
.venv\\Scripts\\activate
```

### 4\. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5\. Run the application

```bash
python -m streamlit run app.py
```

Open the local address displayed in the terminal if the browser does not open automatically. You can also use `run\_app.bat` after configuring the environment.

### Compatibility Note

The existing project documentation states that the saved model was created with scikit-learn 1.6.1. Use the version specified in `requirements.txt` unless the model is retrained and exported with another compatible version.

## How to Use the Application

1. Open the dashboard.
2. Review the dataset information and model evaluation metrics.
3. Navigate to the prediction section.
4. Select an observation or enter the required inputs.
5. Generate a bandwidth prediction.
6. Compare actual and predicted values where available.
7. Explore traffic analysis and model performance visualizations.

The available options depend on the implementation in `app.py`.

## Dataset

The original project documentation identifies the dataset as a 5G network-slicing dataset associated with Zenodo record 21625252.

Before redistributing the dataset, verify its license and attribution requirements.

## Important Notes

* The application uses a saved machine learning model for prediction.
* Model compatibility depends on the Python and library versions used to create and load it.
* Verify reported metrics against the actual evaluation output.
* This application demonstrates machine learning-based network traffic prediction; predictions are not guaranteed real-world network capacity requirements.

## Author

**Ankith G J**

* GitHub: https://github.com/ankithgj31

