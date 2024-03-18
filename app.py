# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:43:15 2024

@author: Admin
"""

import logging
from flask import Flask, render_template, request, jsonify
import joblib

# Configure logging to write to a file
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

app = Flask(__name__)

# Load the trained machine learning model
try:
    model = joblib.load("fish_weight_prediction_model.pkl")
except Exception as e:
    logging.error(f'An error occurred while loading the model: {str(e)}')

# Define the index route to render the frontend
@app.route('/')
def index():
    logging.info('Rendering index.html')
    return render_template('index.html')

# Define a route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the frontend
        data = request.get_json()

        # Extract features from the input data
        length1 = float(data['length1'])
        length2 = float(data['length2'])
        length3 = float(data['length3'])
        height = float(data['height'])
        width = float(data['width'])

        # Make a prediction using the model if it's loaded
        if model:
            prediction = model.predict([[length1, length2, length3, height, width]])

            # Log the prediction
            logging.info(f'Prediction: {prediction}')

            # Return the prediction as JSON
            return jsonify({'weight': prediction[0]})
        else:
            logging.error('Model is not loaded.')
            return jsonify({'error': 'Model is not loaded.'})
    except Exception as e:
        # Log any exceptions that occur during prediction
        logging.error(f'An error occurred during prediction: {str(e)}')
        return jsonify({'error': 'An error occurred during prediction'})

if __name__ == '__main__':
    app.run(debug=True)
