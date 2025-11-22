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
        Age_Group = float(request.form.get("AgeGroup"))

        # Encode Pclass
        if Pclass == "1":
            Pclass_encoded = 0
        elif Pclass == "2":
            Pclass_encoded = 1
        else:
            Pclass_encoded = 2

        # Encode Sex
        Sex_encoded = 1 if Sex == "male" else 0

        # Prepare features for model
        features = np.array([[Pclass_encoded, Sex_encoded, Age, SibSp, Parch, Fare, Embarked, Age_Group]])

        # Prediction
        prediction = model.predict(features)[0]

        # Render result page with all values
        return render_template(
            "result.html",
            Pclass=Pclass,
            Sex=Sex,
            Age=Age,
            SibSp=SibSp,
            Parch=Parch,
            Fare=Fare,
            Embarked=Embarked,
            Age_Group=Age_Group,
            prediction=prediction
        )

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
