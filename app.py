# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 18:22:04 2023

@author: hp
"""

from flask import Flask, render_template, request
import numpy as np
import pickle

model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def predict():
    age = request.form['Age']
    sex = request.form['Sex']
    if sex == 'Male':
        sex = 1
    if sex == 'Female':
        sex = 0
    bmi = request.form['BMI']
    children = request.form['Number of Children']
    if children == 0:
        children = 0
    if children == 1:
        children = 1
    if children == 2:
        children = 2
    if children == 3:
        children = 3
    if children == 4:
        children = 4
    if children == 5:
        children = 5
    smoker = request.form['Smoker']
    if smoker == 'Yes':
        smoker = 1
    if smoker == 'No':
        smoker = 0
    region = request.form['Region']
    if region == 'southeast':
        region = 1
    if region == 'southwest':
        region = 2
    if region == 'northwest':
        region = 3
    if region == 'northeast':
        region = 4
        
    total = [[age,sex,bmi,children,smoker,region]]
    prediction = model.predict(total)
    return render_template('index.html', 
                            prediction_text = "The insurance cost will be {}".format(prediction))

if __name__ == "__main__":
    app.run(debug= True)