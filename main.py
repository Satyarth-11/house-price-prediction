from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    print("Home page visited")
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    sqft = float(request.form["total_sqft"])
    bath = int(request.form["bath"])
    bhk = int(request.form["bhk"])

    print(f"Prediction request → sqft={sqft}, bath={bath}, bhk={bhk}")

    features = np.array([[sqft, bath, bhk]])
    prediction = model.predict(features)

    print(f"Predicted price = {prediction[0]}")

    return render_template("index.html", prediction=prediction[0])


if __name__ == "__main__":
    app.run()