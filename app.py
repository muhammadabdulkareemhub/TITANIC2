from flask import Flask, render_template, request
import pickle
import numpy as np

# Load model
with open("titanic_model.pkl", "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html", survived=None, Age_Group=None)

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

        # Encode Pclass
        if Pclass == "1":
            Pclass_encoded = 0
        elif Pclass == "2":
            Pclass_encoded = 1
        else:
            Pclass_encoded = 2

        # Encode Sex
        Sex_encoded = 1 if Sex == "male" else 0

        # AUTO CALCULATED AGE GROUP
        if Age <= 12:
            Age_Group = 0   # Child
        elif Age <= 19:
            Age_Group = 1   # Teen
        elif Age <= 39:
            Age_Group = 2   # Adult
        elif Age <= 59:
            Age_Group = 3   # Middle-aged
        else:
            Age_Group = 4   # Senior

        # Features to model
        features = np.array([[Pclass_encoded, Sex_encoded, Age, SibSp, Parch, 
                              Fare, Embarked, Age_Group]])

        # Predict
        prediction = model.predict(features)[0]

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

if __name__ == "__main__":
    app.run(debug=True)
