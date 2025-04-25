from flask import Flask, jsonify, request, send_from_directory
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load dataset
data = pd.read_csv('dog_diseases.csv')
X = data.drop('diseases', axis=1)
y = data['diseases']

# Encode target labels
y_encoded = LabelEncoder().fit_transform(y)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y_encoded)

# Save column names and disease labels
all_symptoms = X.columns.tolist()
disease_encoder = LabelEncoder()
disease_encoder.fit(y)

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
    input_vector = {symptom: 1 if symptom in selected_symptoms else 0 for symptom in all_symptoms}
    input_df = pd.DataFrame([input_vector])
    prediction = model.predict(input_df)
    predicted_disease = disease_encoder.inverse_transform(prediction)[0]
    return jsonify({'predicted_disease': predicted_disease})

if __name__ == '__main__':
    app.run(debug=True)
