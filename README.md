#  4project

INDUSTRIAL COPPER MODELING

INTRODUCTION:

This project aims to develop two machine learning models for the copper industry to address the challenges of predicting selling price and lead classification. Manual predictions can be time-consuming and may not result in optimal pricing decisions or accurately capture leads. The models will utilize advanced techniques such as data normalization, outlier detection and handling, handling data in the wrong format, identifying the distribution of features, and leveraging tree-based models, specifically the decision tree algorithm, to predict the selling price and leads accurately.

Regression model details:

The copper industry deals with less complex data related to sales and pricing. However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions. Dealing with these challenges manually can be time-consuming and may not result in optimal pricing decisions. A machine learning regression model can address these issues by utilizing advanced techniques such as data normalization, outlier detection and handling, handling data in wrong format, identifying the distribution of features, and leveraging tree-based models, specifically the decision tree algorithm.

Classification model details:

Another area where the copper industry faces challenges is in capturing the leads. A lead classification model is a system for evaluating and classifying leads based on how likely they are to become a customer. You can use the STATUS variable with WON being considered as Success and LOST being considered as Failure and remove data points other than WON, LOST STATUS values.

ESSENTIAL LIBRARIES:

import warnings, warnings.filterwarnings("ignore"), import pandas as pd, import numpy as py, from sklearn.model_selection import train_test_split, from sklearn.tree import DecisionTreeRegressor, from sklearn.preprocessing import StandardScaler,OneHotEncoder, from sklearn.metrics import mean_squared_error, from sklearn.model_selection, import GridSearchCV, from sklearn.preprocessing import LabelBinarizer, import streamlit as st, from streamlit_option_menu import option_menu, import re

WORK FLOW:

The step by step procedures are shown below;

1. Clone & Preprocessing the Data.

2. Exploring skewness and outliers in the dataset.

3. Transforming the data into a suitable format and performing any necessary cleaning and pre-processing steps.

4. Developing a machine learning regression model which predicts the continuous variable 'Selling_Price' using the decision tree regressor.

5. Developing a machine learning classification model which predicts the Status: WON or LOST using the decision tree classifier.

6. Creating a Streamlit page where you can insert each column value and get the Selling_Price predicted value or Status (Won/Lost).


LESSONS LEARNED:

Python scripting, Data Preprocessing, Machine learning, EDA, Streamlit
