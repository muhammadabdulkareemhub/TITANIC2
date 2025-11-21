from flask import Flask, render_template, request
import pickle
import numpy as np

# Load model
with open("titanic_model.pkl", "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html", survived=None)

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        # Collect form values
        Pclass = request.form.get("Pclass")
        Sex = request.form.get("Sex")
        Age = float(request.form.get("Age"))
        SibSp = float(request.form.get("SibSp"))
        Parch = float(request.form.get("Parch"))
        Fare = float(request.form.get("Fare"))
        Embarked = float(request.form.get("Embarked"))
        Age_Group = float(request.form.get("AgeGroup"))   # Correct field name
