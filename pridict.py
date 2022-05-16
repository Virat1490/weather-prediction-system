from email import header
import pandas as pd
import warnings

import requests
warnings.filterwarnings('ignore')
import numpy as np
from matplotlib import pyplot as plt
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
#import tensorflow as tf
#from tensorflow import keras
from sklearn import preprocessing
import Get_data as gd
import  csv
# from requests.api import head
import requests


#new predictation funtion
def predict_weather():
    #link of API
    url="http://api.thingspeak.com/channels/1735963/feeds.json?result=TRUE"
    headers={
    "Accept":"application/json",
    "Content-Type":"application/json"
    }
    #Get data from API
    response=requests.request("GET",url,headers=headers,data=())
    myjson=response.json()
    # print(myjson)
    #Store the data into Array
    weather_data=[]
    csvheader=['Temperature','Humidity','Pressure','Rain','Intensity','Altitude']
    for i in myjson['feeds']:
        row_insert=[i['field1'],i['field2'],i['field3'],i['field4'],i['field5'],i['field6']]
        weather_data.append(row_insert)
    #open data into CSV files
    with open('weather.csv','w',encoding='UTF8',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(csvheader)
        writer.writerows(weather_data)
    #dataset
    weather_df=pd.read_csv("weather.csv")    
    #Prepare  to trained model
    w=weather_df[list(weather_df.dtypes[weather_df.dtypes!='object'].index)]
    w_y=w.pop('Temperature')
    w_x=w
    # trained model
    train_x,test_x,train_y,test_y=train_test_split(w_x,w_y,test_size=0.03,random_state=4)
    #fit the Model
    reg2=RandomForestRegressor(max_depth=50,random_state=0,n_estimators=100)
    reg2.fit(train_x,train_y)
    #Predict the data
    p2=reg2.predict(test_x)
    #Check for Accuracy
    np.mean((p2-test_y)**2)
    # print("Temperature:")
    # print(p2)
    return p2
    # print(pd.DataFrame({'Actual Value':test_y,'Predicted Value':p2,'Difference':(test_y-p2)}))
    # plt.scatter(test_y,p2)
def predict_humidity():
    #dataset
    weather_df=pd.read_csv("weather.csv")    
    #Prepare  to trained model
    w=weather_df[list(weather_df.dtypes[weather_df.dtypes!='object'].index)]
    w_y=w.pop('Humidity')
    w_x=w
    # trained model
    train_x,test_x,train_y,test_y=train_test_split(w_x,w_y,test_size=0.03,random_state=4)
    #fit the Model
    reg2=RandomForestRegressor(max_depth=50,random_state=0,n_estimators=100)
    reg2.fit(train_x,train_y)
    #Predict the data
    p2=reg2.predict(test_x)
    #Check for Accuracy
    np.mean((p2-test_y)**2)
    # print("humidity")
    # print(p2)
    return p2

predict_weather()
predict_humidity()




   






