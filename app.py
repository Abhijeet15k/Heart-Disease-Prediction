from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)

# Load the trained model
with open('knn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Feature columns from the training data
numerical_cols = ['age', 'resting_blood_pressure', 'serum_cholestoral', 
                 'max_heart_rate', 'oldpeak']
categorical_cols = {
    'chest_pain_type': ['typical', 'atypical', 'non-anginal', 'asymptomatic'],
    'resting_electrocardiographic_results': ['normal', 'ST-T', 'hypertrophy'],
    'ST_segment': ['upsloping', 'flat', 'downsloping'],
    'major_vessels': ['0', '1', '2', '3'],
    'thal': ['normal', 'fixed', 'reversible']
}

# Initialize scaler (values from training data)
scaler = StandardScaler()
scaler.mean_ = np.array([54.366, 131.623, 246.264, 149.647, 1.039])
scaler.scale_ = np.array([9.082, 17.538, 51.831, 22.905, 1.161])

@app.route('/')
def home():
    return render_template('index.html', 
                         numerical_cols=numerical_cols,
                         categorical_cols=categorical_cols)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Create DataFrame with one row
        input_df = pd.DataFrame([data])
        
        # Scale numerical features
        input_df[numerical_cols] = input_df[numerical_cols].astype(float)
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
        
        # One-hot encode categorical features
        for col, values in categorical_cols.items():
            for val in values:
                input_df[f"{col}_{val}"] = (input_df[col] == val).astype(int)
            input_df.drop(col, axis=1, inplace=True)
        
        # Ensure all expected columns are present
        expected_cols = [
            'age', 'resting_blood_pressure', 'serum_cholestoral',
            'max_heart_rate', 'oldpeak',
            'chest_pain_type_typical', 'chest_pain_type_atypical',
            'chest_pain_type_non-anginal', 'chest_pain_type_asymptomatic',
            'resting_electrocardiographic_results_normal',
            'resting_electrocardiographic_results_ST-T',
            'resting_electrocardiographic_results_hypertrophy',
            'ST_segment_upsloping', 'ST_segment_flat', 'ST_segment_downsloping',
            'major_vessels_0', 'major_vessels_1', 'major_vessels_2', 'major_vessels_3',
            'thal_normal', 'thal_fixed', 'thal_reversible'
        ]
        
        # Add missing columns with 0 values
        for col in expected_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns to match training data
        input_df = input_df[expected_cols]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        result = {
            'prediction': int(prediction),
            'probability': float(probability),
            'message': 'Positive for heart disease' if prediction == 1 else 'Negative for heart disease'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
