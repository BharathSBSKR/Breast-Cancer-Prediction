from flask import Flask, render_template, request
import pandas as pd
import joblib

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

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    user_input = {}

    if request.method == "POST":
        try:
            # Get input values from the form
            for feature in selected_features:
                user_input[feature] = float(request.form.get(feature, 0))

            # Create a DataFrame from the input
            input_df = pd.DataFrame([user_input])

            # Make a prediction
            prediction_result = model.predict(input_df)[0]
            prediction = "Malignant (M)" if prediction_result == 1 else "Benign (B)"
        except ValueError:
            prediction = "Invalid input. Please enter numeric values."

    return render_template("index.html", features=selected_features, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
