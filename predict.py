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

# Endpoint to predict disease based on symptoms
@app.route('/predict', methods=['POST'])
def predict_disease():
    try:
        # Get JSON data from request
        input_data = request.get_json()

        # Validate input
        if not isinstance(input_data, list) or len(input_data) != len(X_train.columns):
            return jsonify({"error": "Invalid input. Please provide a list of 132 symptoms."}), 400

        # Create DataFrame from input
        user_input_df = pd.DataFrame([input_data], columns=X_train.columns)

        # Predict disease
        predicted_disease = model.predict(user_input_df)[0]

        return jsonify({"predicted_disease": predicted_disease})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run in development mode
    app.run()
