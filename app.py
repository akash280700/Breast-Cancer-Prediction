import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)

# Load the model from file
filename = "Logistic_Regression_model.pkl"
with open(filename, "rb") as file:
    model = pickle.load(file)

# Define the home page route
@app.route("/")
def home():
    return render_template("index.html")

# Define the prediction route
@app.route("/predict", methods=["POST"])
def predict():
  

    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error', 'fractal dimension error',
       'worst radius', 'worst texture', 'worst perimeter', 'worst area',
       'worst smoothness', 'worst compactness', 'worst concavity',
       'worst concave points', 'worst symmetry', 'worst fractal dimension']
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
        


    # Convert the prediction to a string
    if output == 0:
        result = "Malignant"
    else:
        result = "Benign"

    # Return the prediction as a JSON object
    return render_template('index.html', prediction_text='The breast tumer has {}'.format(result))

if __name__ == "__main__":
    app.run(debug=True)
