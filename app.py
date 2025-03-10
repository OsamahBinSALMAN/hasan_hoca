from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained models
models = {
    "Linear Regression": joblib.load("linear_regression.pkl"),
    "MLP": joblib.load("mlp.pkl"),
    "Random Forest": joblib.load("random_forest.pkl")
}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        min_temp = float(data["min_temp"])
        max_temp = float(data["max_temp"])
        model_name = data["model"]
        class_value = int(data["class"])  # Use class if needed in a future update

        if model_name not in models:
            return jsonify({"error": "Invalid model selection"}), 400

        model = models[model_name]
        X_input = np.array([[min_temp, max_temp]])  # Ensure input shape matches training

        prediction = model.predict(X_input)[0]

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
