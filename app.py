from flask import Flask, render_template, request, jsonify
from sklearn.externals import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('linear_regression_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    input_data = pd.DataFrame(data, index=[0])
    prediction = model.predict(input_data)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
