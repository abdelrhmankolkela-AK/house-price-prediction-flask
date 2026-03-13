from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)

# تحميل الموديل
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

with open(MODEL_PATH, "rb") as f:
    w, b, mean_vals, std_vals = pickle.load(f)


@app.route("/", methods=["GET","POST"])
def home():

    prediction=None
    error_msg=None

    if request.method=="POST":

        try:

            income=float(request.form["income"])
            age=float(request.form["age"])
            rooms=float(request.form["rooms"])
            bedrooms=float(request.form["bedrooms"])
            population=float(request.form["population"])

            if income<=0 or age<=0 or rooms<=0 or bedrooms<=0 or population<=0:
                error_msg="القيم يجب أن تكون أكبر من صفر"

            else:

                data=pd.DataFrame([{
                    "Avg. Area Income":income,
                    "Avg. Area House Age":age,
                    "Avg. Area Number of Rooms":rooms,
                    "Avg. Area Number of Bedrooms":bedrooms,
                    "Area Population":population
                }])

                # نفس feature engineering

                data["Income_per_Pop"]=data["Avg. Area Income"]/data["Area Population"]
                data["Rooms_per_Bed"]=data["Avg. Area Number of Rooms"]/data["Avg. Area Number of Bedrooms"]
                data["Age2"]=data["Avg. Area House Age"]**2

                data["Income_per_Room"]=data["Avg. Area Income"]/data["Avg. Area Number of Rooms"]
                data["Population_per_Room"]=data["Area Population"]/data["Avg. Area Number of Rooms"]
                data["Bedrooms_per_Pop"]=data["Avg. Area Number of Bedrooms"]/data["Area Population"]

                data["Rooms2"]=data["Avg. Area Number of Rooms"]**2
                data["Income2"]=data["Avg. Area Income"]**2
                data["Population2"]=data["Area Population"]**2
                data["Income_Pop"]=data["Avg. Area Income"]*data["Area Population"]

                # normalization

                data=(data-mean_vals)/std_vals

                X=data.values

                prediction = np.dot(X, w) + b
                prediction = prediction.item()

        except:
            error_msg="حدث خطأ في البيانات المدخلة"

    return render_template("home.html",
                           prediction=prediction,
                           error_msg=error_msg)


if __name__=="__main__":
    app.run(debug=True)