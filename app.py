from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained Random Forest model
model = joblib.load("random_forest_model.pkl")

# Define the features
selected_features = [
    'concave points_worst',
    'perimeter_worst',
    'concave points_mean',
    'radius_worst',
    'perimeter_mean',
    'area_worst',
    'radius_mean'
]

# Home route
@app.route('/')
def home():
    return render_template('index.html', features=selected_features)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from form
        user_input = {feature: float(request.form[feature]) for feature in selected_features}
        input_df = pd.DataFrame([user_input])

        # Predict using the model
        prediction = model.predict(input_df)[0]
        result = "Malignant (M)" if prediction == 1 else "Benign (B)"

        return jsonify({
            "success": True,
            "prediction": result,
            "input_features": user_input
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
