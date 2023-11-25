#pip install flask
from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Loading the mlr model
model = pickle.load(open('../training/payments.pkl', 'rb'))
app = Flask(__name__)  # your application


@app.route('/')  # default route
def home():
    return render_template('home.html')  # rendering your home page.

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/submit-form', methods=['POST'])  # prediction route
def predict1():
    '''
    For rendering results on HTML 
    '''
    try:
        # Extracting data from the form
        Step = int(request.form.get('step'))
        Type = int(request.form.get('type'))
        Amount1 = int(request.form.get('amount'))
        Amount2 = int(request.form.get('oldBalanceOrg'))
        Amount3 = int(request.form.get('newBalanceOrg'))
        CardNumber1 = int(request.form.get('oldBalanceDest'))
        CardNumber2 = int(request.form.get('newBalanceDest'))

        # Create a DataFrame from the form data
        input_df = pd.DataFrame({
        'step': [Step],
        'type': [Type],
        'amount': [Amount1],
        'oldbalanceOrg': [Amount2],
        'newbalanceOrig': [Amount3],
        'oldbalanceDest': [CardNumber1],
        'newbalanceDest': [CardNumber2]
    })

        # Make prediction using the pre-trained model
        prediction = model.predict(input_df)
        result_str = str(prediction[0])

        return render_template("submit.html", result="  " + result_str + "!")
    except Exception as e:
        return render_template("submit.html", result="Error: " + str(e))

# running your application
if __name__ == "__main__":
    app.run()
