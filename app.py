from flask import Flask , render_template , request
import pickle
app=Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")

with open("models/house_price.pkl", mode="rb") as file:
    Linreg=pickle.load(file)

@app.route("/house",methods=["GET","POST"])
def house():
    yp = None

    if request.method=="POST":
        aai  = int(request.form.get("v1"))
        aaha = int(request.form.get("v2"))
        aanr = int(request.form.get("v3"))
        aanb = int(request.form.get("v4"))
        ap = int(request.form.get("v5"))

        yp= int(Linreg.predict([[aai, aaha, aanr, aanb, ap]])[0])
    

    return render_template("house_price.html" , price=yp)

with open("models/custchurn.pkl", mode="rb") as file:
    Linreg2=pickle.load(file)

with open("models/custchurn_encoder.pkl", mode="rb") as file:
    encoder=pickle.load(file)

@app.route("/churn" ,methods=["GET" , "POST"])
def prediction():
    result = None

    if request.method=="POST":
        c1=request.form.get("v1")
        c2=int(request.form.get("v2"))
        c3=request.form.get("v3")
        c4=request.form.get("v4")
        c5=int(request.form.get("v5"))
        c6=request.form.get("v6")
        c7=request.form.get("v7")
        c8=request.form.get("v8")
        c9=request.form.get("v9")
        c10=request.form.get("v10")
        c11=request.form.get("v11")
        c12=request.form.get("v12")
        c13=request.form.get("v13")
        c14=request.form.get("v14")
        c15=request.form.get("v15")
        c16=request.form.get("v16")
        c17=request.form.get("v17")
        c18=float(request.form.get("v18"))
        c19=float(request.form.get("v19"))

        cat_data = [[c1,c3,c4,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17]]
        cat_data = encoder.transform(cat_data)
        final_data = [[
            cat_data[0][0],   # gender
            c2,               # SeniorCitizen
            cat_data[0][1],   # Partner
            cat_data[0][2],   # Dependents
            c5,               # tenure

            cat_data[0][3],   # PhoneService
            cat_data[0][4],   # MultipleLines
            cat_data[0][5],   # InternetService
            cat_data[0][6],   # OnlineSecurity
            cat_data[0][7],   # OnlineBackup

            cat_data[0][8],   # DeviceProtection
            cat_data[0][9],   # TechSupport
            cat_data[0][10],  # StreamingTV
            cat_data[0][11],  # StreamingMovies
            cat_data[0][12],  # Contract

            cat_data[0][13],  # PaperlessBilling
            cat_data[0][14],  # PaymentMethod

            c18,              # MonthlyCharges
            c19               # TotalCharges
            ]]
        yp1= Linreg2.predict(final_data)[0]
        if yp1== 1:
            result = "The customer is likely to churn."
        else:
            result = "The customer is not likely to churn."


    return render_template("customer_churn.html" , sales=result)


with open("models/diabetes.pkl", "rb") as file:
    diabetes_model = pickle.load(file)


@app.route("/diabetes", methods=["GET", "POST"])
def diabetes():

    result = None

    if request.method == "POST":

        p = int(request.form.get("v1"))
        g = int(request.form.get("v2"))
        bp = int(request.form.get("v3"))
        st = int(request.form.get("v4"))
        ins = int(request.form.get("v5"))
        bmi = float(request.form.get("v6"))
        dpf = float(request.form.get("v7"))
        age = int(request.form.get("v8"))

        pred = diabetes_model.predict([[p, g, bp, st, ins, bmi, dpf, age]])[0]

        if pred == 1:
            result = "Diabetes Detected ⚠️"
        else:
            result = "No Diabetes Detected ✅"

    return render_template( "diabetes.html", result=result )

with open("models/trainedmodel.pkl", "rb") as file:
    bigmart_model = pickle.load(file)


@app.route("/bigmart", methods=["GET", "POST"])
def bigmart():

    sales = None

    if request.method == "POST":

        a = float(request.form.get("v1"))
        b = float(request.form.get("v2"))
        c = float(request.form.get("v3"))
        d = float(request.form.get("v4"))
        e = float(request.form.get("v5"))
        f = float(request.form.get("v6"))
        g = float(request.form.get("v7"))
        h = int(request.form.get("v8"))
        i = float(request.form.get("v9"))
        j = float(request.form.get("v10"))
        k = float(request.form.get("v11"))

        sales = bigmart_model.predict([[a,b,c,d,e,f,g,h,i,j,k]])[0]

    return render_template("bigmart.html",sales=sales)

with open("models/addmission.pkl", "rb") as file:
    admission_model = pickle.load(file)


@app.route("/admission", methods=["GET", "POST"])
def admission():

    result = None

    if request.method == "POST":

        a = int(request.form.get("v1"))
        b = int(request.form.get("v2"))
        c = int(request.form.get("v3"))
        d = float(request.form.get("v4"))
        e = float(request.form.get("v5"))
        f = float(request.form.get("v6"))
        g = int(request.form.get("v7"))

        pred = admission_model.predict([[a,b,c,d,e,f,g]])[0]

        result = round(float(pred), 2)

    return render_template("admission.html",result=result)

if __name__ == "__main__":
    app.run(debug=True)

