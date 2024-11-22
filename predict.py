from flask import Flask, request, jsonify
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Initialize Flask app
app = Flask(__name__)

# Load training data
train_data = pd.read_csv('Training.csv')

# Separate features (symptoms) and target (prognosis)
X_train = train_data.drop(columns=['prognosis'])
y_train = train_data['prognosis']

# Initialize and train the Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Endpoint for predicting disease
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.json
        
        # Ensure the JSON contains all necessary symptoms
        if not all(symptom in data for symptom in X_train.columns):
            return jsonify({"error": "Missing symptoms in input"}), 400

        # Create a DataFrame for prediction
        user_input_df = pd.DataFrame([data], columns=X_train.columns)
        
        # Predict the disease
        predicted_disease = model.predict(user_input_df)[0]
        
        return jsonify({"disease": predicted_disease}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
