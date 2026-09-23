from flask import Flask, jsonify, request, send_from_directory
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)

# Load the dataset and model
data = pd.read_csv('dog_diseases.csv')
X = data.drop('diseases', axis=1)

# Load pre-trained model
model = joblib.load('dog_disease_model.pkl')

# Encode the target labels the same way as before
label_encoder = LabelEncoder()
y = data['diseases']
label_encoder.fit(y)

# Save column names and disease labels
all_symptoms = X.columns.tolist()
disease_encoder = label_encoder

# Compute common symptoms based on frequency
symptom_frequency = X.sum().sort_values(ascending=False)
common_symptoms = symptom_frequency.head(5).index.tolist()

@app.route('/')
def serve_index():
    return send_from_directory('templates', 'index.html')

@app.route('/get_initial_symptoms', methods=['GET'])
def get_initial_symptoms():
    return jsonify({'symptoms': common_symptoms})

@app.route('/get_filtered_symptoms', methods=['POST'])
def get_filtered_symptoms():
    selected_symptoms = request.json['selected_symptoms']

    # Get subset of data where all selected symptoms are present
    subset = data.copy()
    for symptom in selected_symptoms:
        if symptom in X.columns:
            subset = subset[subset[symptom] == 1]

    # Get relevant symptoms remaining in the subset
    remaining_symptoms = []
    if not subset.empty:
        for col in X.columns:
            if col not in selected_symptoms and subset[col].sum() > 0:
                remaining_symptoms.append(col)

    return jsonify({'remaining_symptoms': remaining_symptoms})

@app.route('/predict', methods=['POST'])
def predict():
    selected_symptoms = request.json['symptoms']

    # Check if there are no symptoms selected
    if not selected_symptoms:
        return jsonify({'error': 'No symptoms selected. Please select at least one symptom.'}), 400

    input_vector = {symptom: 1 if symptom in selected_symptoms else 0 for symptom in all_symptoms}
    input_df = pd.DataFrame([input_vector])

    # Predict probabilities
    probabilities = model.predict_proba(input_df)[0]

    # Get top 3 predictions
    top_indices = np.argsort(probabilities)[::-1][:3]
    top_diseases = disease_encoder.inverse_transform(top_indices)
    top_confidences = probabilities[top_indices]

    results = []
    for disease, confidence in zip(top_diseases, top_confidences):
        results.append({
            'disease': disease,
            'chance_percent': round(confidence * 100, 2)
        })

    return jsonify({'predictions': results})

if __name__ == '__main__':
    app.run(debug=True)

