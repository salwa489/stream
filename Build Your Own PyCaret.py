import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
import pycaret
from pycaret.classification import setup as classification_setup, compare_models as classification_compare_models
from pycaret.regression import setup as regression_setup, compare_models as regression_compare_models

import streamlit as st

def load_data(file_path):
    _, file_extension = file_path.rsplit('.', 1)

    if file_extension == 'csv':
        return pd.read_csv(file_path)
    elif file_extension == 'xlsx':
        return pd.read_excel(file_path)
    elif file_extension == 'sql':
        conn = sqlite3.connect('database.db')
        # Read data from the database
        # ...
        pass
    else:
        raise ValueError('Unsupported file format.')

def handle_duplicate_values(data):
    # Remove duplicate rows
    data.drop_duplicates(inplace=True)
    return data

def generate_histograms(data):
    for col in data.select_dtypes(include='number'):
        plt.figure()
        fig, ax = plt.subplots()
        ax = sns.histplot(data[col])
        plt.title(f'Histogram of {col}')
        st.pyplot(fig)

def generate_box_plots(data):
    for col in data.select_dtypes(include='number'):
        plt.figure()
        fig, ax = plt.subplots()
        ax = sns.boxplot(data=data[col])
        plt.title(f'Box Plot of {col}')
        st.pyplot(fig)

def generate_scatter_plots(data):
    numerical_cols = data.select_dtypes(include='number').columns

    for i, col1 in enumerate(numerical_cols):
        for j, col2 in enumerate(numerical_cols):
            if i < j:
                plt.figure()
                fig, ax = plt.subplots()
                ax = sns.scatterplot(data=data, x=col1, y=col2)
                plt.title(f'Scatter Plot of {col1} vs {col2}')
                st.pyplot(fig)

def handle_normalize_missing_values(data, categorical_features_tdeal, continuous_features_tdeal):
    continuous_features = []
    categorical_features = []

    # Identify the type of each feature in the data
    for column in data.columns:
        if data[column].dtype == 'object' or data[column].nunique() <= 10:
            categorical_features.append(column)
        else:
            continuous_features.append(column)

    # Fill missing values in categorical features
    if categorical_features_tdeal == 'ordinal_encoder':
        ordinal_encoder = OrdinalEncoder()
        for feature in categorical_features:
            data[feature] = ordinal_encoder.fit_transform(data[feature].values.reshape(-1, 1))
    elif categorical_features_tdeal == 'imputer':
        imputer = SimpleImputer(strategy='most_frequent')
        for feature in categorical_features:
            data[feature] = imputer.fit_transform(data[feature].values.reshape(-1, 1))

    # Fill missing values in continuous features
    if continuous_features_tdeal == 'mean()':
        for feature in continuous_features:
            data[feature].fillna(data[feature].mean(), inplace=True)
    elif continuous_features_tdeal == 'median()':
        for feature in continuous_features:
            data[feature].fillna(data[feature].median(), inplace=True)
    elif continuous_features_tdeal == 'mode()':
        for feature in continuous_features:
            data[feature].fillna(data[feature].mode()[0], inplace=True)

    # Normalize continuous features
    scaler = MinMaxScaler()
    data[continuous_features] = scaler.fit_transform(data[continuous_features])

    # Encode categorical features
    encoder = LabelEncoder()
    for feature in categorical_features:
        data[feature] = encoder.fit_transform(data[feature])

    return data

def train_validate_models(data, target_variable):
    continuous_features = []
    categorical_features = []
    for column in data.columns:
        if data[column].dtype == 'object' or data[column].nunique() <= 10:
            categorical_features.append(column)
        else:
            continuous_features.append(column)

    if target_variable in categorical_features:
        try:
            print('The case is classification')
            classification_setup(data=data, target=target_variable)
            classification_compare_models()
        except Exception as e:
            print(f"An error occurred during classification model training: {str(e)}")
    elif target_variable in continuous_features:
        try:
            print('The case is regression')
            regression_setup(data=data, target=target_variable)
            regression_compare_models()
        exceptException as e:
            print(f"An error occurred during regression model training: {str(e)}")

# Streamlit code
def main():
    st.title("Data Analysis and Modeling")
    st.sidebar.title("Options")

    # Upload file
    uploaded_file = st.sidebar.file_uploader("Upload File", type=["csv", "xlsx", "sql"])

    if uploaded_file is not None:
        # Load data
        data = load_data(uploaded_file.name)

        # Handle duplicate values
        duplicate_values = data.duplicated().sum()
        if duplicate_values > 0:
            data = handle_duplicate_values(data)

        # Sidebar options
        analysis_options = st.sidebar.checkbox("Data Analysis")
        modeling_options = st.sidebar.checkbox("Modeling")

        if analysis_options:
            st.subheader("Data Analysis")
            # Generate histograms
            if st.checkbox("Generate Histograms"):
                generate_histograms(data)

            # Generate box plots
            if st.checkbox("Generate Box Plots"):
                generate_box_plots(data)

            # Generate scatter plots
            if st.checkbox("Generate Scatter Plots"):
                generate_scatter_plots(data)

        if modeling_options:
            st.subheader("Modeling")
            # Target variable selection
            target_variable = st.selectbox("Select Target Variable", data.columns)

            # Handle missing values and normalization
            categorical_features_tdeal = st.selectbox("Categorical Features Treatment", ["ordinal_encoder", "imputer"])
            continuous_features_tdeal = st.selectbox("Continuous Features Treatment", ["mean()", "median()", "mode()"])
            data = handle_normalize_missing_values(data, categorical_features_tdeal, continuous_features_tdeal)

            # Train and validate models
            train_validate_models(data, target_variable)

if __name__ == "__main__":
    main()
