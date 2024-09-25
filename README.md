# End-to-End Employee Burnout Predictor

This repository contains the code for building an Employee Burnout Prediction model using Scikit-learn for machine learning, FastAPI for API creation, and Gradio for the user interface.

## Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Running the Project](#running-the-project)
- [Next Steps](#next-steps)

## Project Overview
The **Employee Burnout Predictor** project aims to predict burnout levels in employees based on provided features. The workflow includes data preprocessing, model training, API creation, and a user-friendly interface to allow real-time predictions.

## Project Structure
- **`data_cleaning.py`**: Preprocesses the raw training data (`Employee_Burnout_Train.csv`) and outputs a cleaned dataset (`Cleaned_ML_Dataset.csv`).
  
- **`model_io.py`**: Contains `save_model` and `load_model` functions for saving and loading machine learning models.

- **`model_building.py`**: Initializes different models, evaluates their performance, and uses Grid Search to optimize hyperparameters. The best-performing model (Random Forest) is saved as `random_forest_model.pkl`.

- **`api.py`**: Loads the trained model and provides a FastAPI endpoint (`/prediction_endpoint`) to receive user inputs and return predictions.

- **`app.py`**: Creates a Gradio interface that interacts with the FastAPI endpoint, allowing users to input data and get burnout predictions.

## Running the Project

### Step 1: Data Cleaning
First, you need to run the `data_cleaning.py` script. This script will preprocess the provided training data (`Employee_Burnout_Train.csv`) and generate a cleaned CSV file named `Cleaned_ML_Dataset.csv`.

### Step 2: Model I/O Functions
Next, run the `model_io.py` script to define the `save_model` and `load_model` functions. These functions will be used later to save and retrieve the trained machine learning model.

### Step 3: Model Building and Training
Run the `model_building.py` script to initialize different models, including Random Forest, and evaluate their performance. After determining the best model, perform a Grid Search to optimize hyperparameters. Once the model is trained and evaluated, save it as `random_forest_model.pkl` using the `save_model` function.

### Step 4: FastAPI Server
Once the model is saved, run the `api.py` script. This will load the saved model and create a FastAPI endpoint at `/prediction_endpoint`. The endpoint allows users to send input data, which is then processed, and the predicted burnout level is returned.

### Step 5: Gradio Interface
Finally, run the `app.py` script. This will create a Gradio interface that connects with the FastAPI endpoint. Ensure the FastAPI server is running in the background. The Gradio interface will allow users to input data and receive burnout predictions directly.
