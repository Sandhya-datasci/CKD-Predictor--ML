from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the saved best model (no scaler used)
model = pickle.load(open('CKD.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/Prediction', methods=['POST', 'GET'])
def predict_form():
    return render_template('indexnew.html')

@app.route('/Home', methods=['POST', 'GET'])
def my_home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Reading inputs from user form
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    features_name = ['blood_urea', 'blood_glucose_random', 'anemia',
                     'coronary_artery_disease', 'pus_cell', 'red_blood_cells',
                     'diabetesmellitus', 'pedal_edema']
    
    df = pd.DataFrame(features_value, columns=features_name)
    
    # Prediction
    output = model.predict(df)
    result = "CKD Detected" if output[0] == 1 else "No CKD Detected"
    
    return render_template('result.html', prediction_text=result)


if __name__ == '__main__':
    app.run(debug=True)
