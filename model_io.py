#Importing packages

import joblib

# Function to save model

def save_model(model):
    joblib.dump(model, 'random_forest_model.pkl')

def load_model():
    model = joblib.load('random_forest_model.pkl')
    return model
