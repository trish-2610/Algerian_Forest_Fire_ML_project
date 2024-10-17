from flask import Flask , request , render_template , jsonify
import numpy as np 
import pandas as pd 
import pickle 
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application 

## importing ridge regressor and Standard Scaler pickle 
ridge_model = pickle.load(open("Models/ridge.pkl","rb"))
standard_scaler = pickle.load(open("Models/scaler.pkl","rb"))

@app.route("/")  ## home page 
def index():
    return render_template("index.html")

@app.route("/predict_data", methods=["GET", "POST"])
def predict_data():
    if request.method == "POST":
        # Extract form data as floats
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        WS = float(request.form.get("WS"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        # Standardization (Note: Use the variables instead of strings)
        new_scaled_data = standard_scaler.transform([[Temperature, RH, WS, Rain, FFMC, DMC, ISI, Classes, Region]])

        # Prediction
        result = ridge_model.predict(new_scaled_data)  # The result will be a list
        
        # Pass the result to the template (result[0] gets the first item of the list)
        return render_template("home.html", results=result[0])

    # In case the method is GET
    return render_template("home.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0")