from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained models and feature names
with open('trained_models.pkl', 'rb') as f:
    trained_models = pickle.load(f)

with open('features.pkl', 'rb') as f:
    feature_names = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        input_data = request.get_json()
        if not input_data:
            return jsonify({"error": "Invalid input. JSON body is required."}), 400

        # Convert input data to DataFrame
        input_df = pd.DataFrame.from_dict(input_data, orient='index')

        # Check for missing features
        missing_features = [f for f in feature_names if f not in input_df.columns]
        if missing_features:
            return jsonify({
                "error": "Missing features.",
                "expected_features": feature_names,
                "missing_features": missing_features,
                "message": "Ensure your input includes all required features."
            }), 400

        # Add missing columns with default values
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # Response dictionary
        response = {}

        # Iterate over samples and make predictions
        for index, sample in input_df.iterrows():
            sample_predictions = {}
            for model_name, model in trained_models.items():
                try:
                    # Predict probabilities or output values
                    if hasattr(model, "predict_proba"):
                        prediction = model.predict_proba(sample.values.reshape(1, -1))
                        sample_predictions[model_name] = float(np.max(prediction))  
                    else:
                        prediction = model.predict(sample.values.reshape(1, -1))
                        sample_predictions[model_name] = float(prediction[0])  
                except Exception as e:
                    sample_predictions[model_name] = f"Error: {str(e)}"

            response[index] = sample_predictions

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run()
