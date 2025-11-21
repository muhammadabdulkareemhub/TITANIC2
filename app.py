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
    try:
        # Collect form values
        Pclass = request.form.get("Pclass")
        Sex = request.form.get("Sex")
        Age = float(request.form.get("Age"))
        SibSp = float(request.form.get("SibSp"))
        Parch = float(request.form.get("Parch"))
        Fare = float(request.form.get("Fare"))
        Embarked = float(request.form.get("Embarked"))
        Age_Group = float(request.form.get("AgeGroup"))   # FIXED name

        # Encode Pclass
        if Pclass == "1":
            Pclass = 0
        elif Pclass == "2":
            Pclass = 1
        else:
            Pclass = 2

        # Encode Sex
        Sex = 1 if Sex == "male" else 0

        # Build feature array
        features = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Age_Group]])

        # Model prediction
        prediction = model.predict(features)[0]

        # Convert prediction to True/False for HTML
        survived = True if prediction == 1 else False

        return render_template("index.html", survived=survived)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
