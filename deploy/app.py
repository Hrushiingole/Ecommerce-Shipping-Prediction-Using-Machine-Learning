from flask import Flask, render_template, request
import joblib
import pickle
import pandas as pd
import numpy as np
with open('model_pickle','rb') as f:
    model=pickle.load(f)
sc=pickle.load(open("modelscaler.pkl","rb"))
le = joblib.load("labelen")
               
app = Flask(__name__)
@app.route('/')
def loadpage():
    return render_template("index.html")


@app.route('/y_predict', methods = ["POST"])
def prediction():
    Warehouse_block=request.form['Warehouse_block']
    
    if(Warehouse_block=='A'):
            Warehouse_block=0
    
    if(Warehouse_block=='B'):
            Warehouse_block=1
    
    if(Warehouse_block=='C'):
            Warehouse_block=2
    
    if(Warehouse_block=='D'):
            Warehouse_block=3
    
    if(Warehouse_block=='F'):
            Warehouse_block=4
    
    
    Mode_of_Shipment1=request.form['Mode_of_Shipment']
    if(Mode_of_Shipment1=='Flight'):
        Mode_of_Shipment=0    
    
    if(Mode_of_Shipment1=='Ship'):
        Mode_of_Shipment=1    
    
    if(Mode_of_Shipment1=='Road'):
        Mode_of_Shipment=2    
    
    Product_importance1=request.form['Product_importance']
    if(Product_importance1=='low'):
            Product_importance=0
    
    if(Product_importance1=='medium'):
            Product_importance=1
    
    if(Product_importance1=='high'):
            Product_importance=2
    
    
    
    

    Customer_care_calls=request.form["Customer_care_calls"]
    Customer_rating=request.form["Customer_rating"]
    Cost_of_the_Product=request.form["Cost_of_the_Product"]
    prior_purchase=request.form["prior_purchases"]
    
    
    Discount_offered=request.form["Discount_offered"]
    Weight_in_gms=request.form["Weight_in_gms"]
    x_test=[[Warehouse_block,Mode_of_Shipment,Customer_care_calls,Customer_rating,Cost_of_the_Product,prior_purchase,Product_importance,Discount_offered,Weight_in_gms]]
    
    p=np.array(sc.transform(x_test))
    prediction=model.predict(p)
    
    if(prediction==1):
        text='order will be delivered on time'
    else:
        text='order will not be delivered on time'
    return render_template("index.html", prediction_text=text)


if __name__ == "__main__":
    app.run(debug = True)
