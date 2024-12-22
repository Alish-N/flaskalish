# app.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Configure CORS
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:5500",
            "http://127.0.0.1:5500",
            "http://localhost:5000",
            "http://127.0.0.1:5000"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Load the saved models and preprocessors
try:
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    le_dict = joblib.load('models/label_encoders.pkl')
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise

def preprocess_input(data):
    """
    Preprocesses input data for prediction
    """
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # Feature Engineering
        input_df['price_per_mb'] = input_df['monthly_charges'] / input_df['bandwidth_mb']
        input_df['usage_efficiency'] = input_df['avg_monthly_gb_usage'] / input_df['bandwidth_mb']
        input_df['satisfaction_index'] = (input_df['customer_rating'] * 10) / (1 + input_df['support_tickets_opened'])
        input_df['support_ticket_ratio'] = input_df['support_tickets_opened'] / input_df['customer_rating']
        input_df['value_score'] = (input_df['bandwidth_mb'] * input_df['customer_rating']) / input_df['monthly_charges']
        input_df['usage_intensity'] = input_df['avg_monthly_gb_usage'] / input_df['bandwidth_mb']
        input_df['efficiency_rating'] = input_df['customer_rating'] * input_df['usage_efficiency']
        input_df['cost_per_rating'] = input_df['monthly_charges'] / input_df['customer_rating']
        
        # Encode categorical variables
        categorical_cols = ['service_plan', 'connection_type']
        for col in categorical_cols:
            input_df[col] = le_dict[col].transform(input_df[col])
        
        # Ensure features are in the exact same order as during training
        features = [
            'service_plan', 
            'connection_type', 
            'monthly_charges', 
            'bandwidth_mb',
            'avg_monthly_gb_usage', 
            'customer_rating',
            'support_tickets_opened', 
            'price_per_mb',
            'usage_efficiency', 
            'satisfaction_index',
            'support_ticket_ratio', 
            'value_score',
            'usage_intensity', 
            'efficiency_rating',
            'cost_per_rating'
        ]
        
        # Create DataFrame with features in correct order
        X = input_df[features].copy()
        
        # Print feature names for debugging
        print("Features being used:", X.columns.tolist())
        
        # Transform the features
        X_scaled = scaler.transform(X)
        
        return X_scaled
        
    except Exception as e:
        print(f"Detailed error: {str(e)}")
        raise ValueError(f"Error in preprocessing input: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        
        # Preprocess input
        X_scaled = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        
        # Prepare response
        response = {
            'churn_risk': str(prediction),
            'probability': {
                'Low': float(probabilities[0]),
                'Medium': float(probabilities[1]),
                'High': float(probabilities[2])
            },
            'confidence': float(max(probabilities))
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

@app.route('/')
def home():
    return send_file('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)