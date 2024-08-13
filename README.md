# Breast-Cancer-Analysis
# Breast Cancer Data Analysis and Streamlit App

This project involves the analysis of the Breast Cancer dataset, implementation of an Artificial Neural Network (ANN) model, and deployment as an interactive web app using Streamlit.

## Project Structure
- Script for data loading and preprocessing.
- Script for feature selection using SelectKBest.
- Script for hyperparameter tuning using Grid Search CV.
- Script for training and evaluating the ANN model.
- `app.py`: Streamlit app for user interaction and predictions.

## Setup
1. Clone the repository.
2. Set up a virtual environment.
3. Install the required packages using `pip install -r requirements.txt`.
4. Run the Streamlit app with `streamlit run app.py`.

## Dataset
The Breast Cancer dataset used in this project can be downloaded from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29).

## Instructions
1. Run the scripts sequentially to preprocess data, select features, and train the model.
2. Use the Streamlit app (`app.py`) to interact with the dataset and view predictions.
3. All results, including model evaluation and selected features, are saved as CSV or text files.

## Requirements
- tensorflow==2.15.0
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- streamlit
- scikeras
