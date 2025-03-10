from flask import Flask, request, jsonify
from flask_cors import CORS 
import joblib
import numpy as np
import warnings
 
warnings.filterwarnings('ignore')
app = Flask(__name__)
CORS(app)

# Load trained models
models = {
    "Linear Regression": joblib.load("Linear_Regression_model.pkl"),
    "MLP": joblib.load("MLP_model.pkl"),
    "Random Forest": joblib.load("Random_Forest_model.pkl")
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
        X_input = np.array([[min_temp, max_temp,class_value]])  # Ensure input shape matches training

        prediction = model.predict(X_input)[0]

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
