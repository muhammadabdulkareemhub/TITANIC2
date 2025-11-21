from flask import Flask, render_template, request
import pickle
import numpy as np

# Load model
with open("titanic_model.pkl", "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/submit', methods=['POST'])
def submit():
    # Collect form values
    Pclass = request.form['Pclass']
    Sex = request.form['Sex']
    Age = float(request.form['Age'])
    SibSp = float(request.form['SibSp'])
    Parch = float(request.form['Parch'])
    Fare = float(request.form['Fare'])
    Embarked = float(request.form['Embarked'])
    Age_Group = float(request.form['Age Group'])

    # Encode Pclass
    if Pclass == "1":
        Pclass_encoded = 0
    elif Pclass == "2":
        Pclass_encoded = 1
    else:
        Pclass_encoded = 2

    # Encode Sex
    Sex_encoded = 1 if Sex == "male" else 0

    # Build feature array
    features = np.array([[Pclass_encoded, Sex_encoded, Age, SibSp, Parch, Fare, Embarked, Age_Group]])

    # Model prediction
    prediction = model.predict(features)

    return render_template(
        "result.html",
        Pclass=Pclass,          # original value for display
        Sex=Sex,                # original sex (male/female)
        Age=Age,
        SibSp=SibSp,
        Parch=Parch,
        Fare=Fare,
        Embarked=Embarked,
        Age_Group=Age_Group,
        prediction=prediction[0]
    )

if __name__ == '__main__':
    app.run(debug=True)
