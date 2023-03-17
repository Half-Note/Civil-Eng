# ------------------------------------------------------
# ---------------------- main.py -----------------------
# ------------------------------------------------------

from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
import numpy as np
import pandas as pd
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import datetime as dt
import math
from weatherbit.api import Api
import json
import urllib
import scipy
from sklearn.metrics import _pairwise_distances_reduction
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import sys
import os


fetchdata_status=0
maturity_status=0
strength_status=0
weather_status =0
rg_strength_status=0
train_model_status=0
temp_train_status = 0
mat_weather_status =0

class MyDialog(QDialog):
    def __init__(self):
        super().__init__()
        loadUi('about.ui', self)  # Load the UI file

class MyDialog1(QDialog):
    def __init__(self):
        super().__init__()
        loadUi('user.ui', self)  # Load the UI file

class MatplotlibWidget(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        loadUi("qt_designer.ui", self)
        self.setWindowTitle("Maturity IoT Project - ML Integrated approach for Monitoring Concrete Condition")
        self.pushButton_fetch_data.clicked.connect(self.fetch_data)
        self.pushButton_temptime_graph.clicked.connect(self.realtime_temptime_graph)
        self.pushButton_weather.clicked.connect(self.weather)
        self.pushButton_maturity.clicked.connect(self.maturity)
        self.pushButton_maturity_strength.clicked.connect(self.strength)
        self.pushButton_Str_Train.clicked.connect(self.Str_Train)
        self.pushButton_Train_use.clicked.connect(self.Train_use)
        self.pushButton_Mat_Strength_Graph.clicked.connect(self.Mat_strength_graph)
        self.pushButton_mat_weather_Graph.clicked.connect(self.mat_weather)
        self.pushButton_mat_weather_ml.clicked.connect(self.mat_weather_ml)
        self.pushButton_mat_weather_ml_str.clicked.connect(self.mat_weather_ml_str)
        self.addToolBar(NavigationToolbar(self.MplWidget_2.canvas, self))
        self.lineEdit_strength_input.setText("0.0")
        self.sensor_id_input.setText("T01")
        self.mix_design_id_input.setText("S_Temp_32")
        self.lineEdit_label_Train_S_input.setText("T01")
        self.lineEdit_Train_M_input.setText("S_Temp_32")
        self.day1input.setText("2")
        self.day3input.setText("6")
        self.day7input.setText("12")
        self.day14input.setText("16")
        self.day28input.setText("20")
        self.datumtemp_input.setText("0.0")
        self.actionAbout_soft.triggered.connect(self.about)
        self.actionuser.triggered.connect(self.user)

        cred = credentials.Certificate('secret key.json')
        firebase_admin.initialize_app(cred, {
            'databaseURL': "https://iot-research######2-default-rtdb.asia-southeast1.firebasedatabase.app"
        })


    def about(self):
        dialog = MyDialog()
        dialog.exec_()
    def user(self):
        dialog = MyDialog1()
        dialog.exec_()



    def fetch_data(self):
        # data loading
        # Connecting and retrieving data from the database
        # Fetch the service account key JSON file contents
        global mix_design_id
        sensor_id = str(self.sensor_id_input.text())
        mix_design_id = str(self.mix_design_id_input.text())
        full_id = "/SensorData/lwgbIPQ6lpPSuE9xUhZIqsj3Wsi1/" + sensor_id
        # Initialize the app with a service account, granting admin privileges
        ref = db.reference(full_id).get()

        if type(ref) is dict:
            # Create a data frame from dictionery returned by firebase
            sensor_dataframe = pd.DataFrame.from_dict(ref.values())
            sensor_dataframe = sensor_dataframe.apply(pd.to_numeric)
            # Convert the data and time to human readable form
            timestamp_array = sensor_dataframe["timestamp"].to_numpy()
            timestamp_array = timestamp_array.astype(str)
            print(timestamp_array)
            i = 0
            for time in timestamp_array:
                timestamp_array[i] = dt.datetime.fromtimestamp(int(time)).isoformat("#", "minutes")
                i = i + 1
            sensor_dataframe['Time'] = timestamp_array
            sensor_dataframe.drop('timestamp', inplace=True, axis=1)
            print('work')
            global sensor_dataframe_g
            sensor_dataframe_g = sensor_dataframe.copy()
            sum2=sensor_dataframe_g[sensor_dataframe_g.columns[0]].count()
            self.lineEdit_timeelapsed.setText(str(round(sum2/48,0)))
            self.statusBar().showMessage("Data Fetched Successfully!!")
            global fetchdata_status
            fetchdata_status=1
        else:
            self.statusBar().showMessage("Sensor ID not Found!!")

    def maturity(self):
        print(fetchdata_status)
            global sum1
            global datum_temp
            datum_temp = float(self.datumtemp_input.text())
            maturity_df = sensor_dataframe_g.copy()
            series_size = maturity_df['Time'].size
            maturity_df['Time'] = np.arange(0, series_size / 2, 0.5)
            sum1 = 0.0
            for x, y in maturity_df.items():
                if x == mix_design_id:
                        maturity_df[x][j] = sum1 / 24
                    sum1 = 0.0
            global maturity_df_g
            maturity_df_g=maturity_df.copy()
            self.lineEdit_maturity.setText(str(sum1))
            ax = maturity_df_g['Time']
            ay = maturity_df_g[mix_design_id]
            xtick_location = ax[::48]
            xtick_labels = [x for x in ax[::48]]
            self.MplWidget_2.canvas.axes.clear()
            self.MplWidget_2.canvas.axes.plot(ax, ay, color='tab:orange',label='Saul Maturity Index °C_Hr')
            self.MplWidget_2.canvas.axes.set_title('Maturity Graph')
            self.MplWidget_2.canvas.axes.set_xlabel('Time Hr')
            self.MplWidget_2.canvas.axes.set_ylabel('Maturity °C-Hr')
            self.MplWidget_2.canvas.axes.legend(['Saul_Maturity_Index_°C_Hr'],loc='lower right')
            self.MplWidget_2.canvas.axes.set_xticks(xtick_location, xtick_labels, rotation=45)
            self.MplWidget_2.canvas.draw()
            self.lineEdit_maturity.setText(str(round(maturity_df_g[mix_design_id].values[-1],0)))

            self.statusBar().showMessage("Maturity calculated & Graphed Successfully!!")
            global maturity_status
            maturity_status = 1

    def strength(self):
        if maturity_status !=1:
            self.maturity()
        Lab_Strength = float(self.lineEdit_strength_input.text())
        if fetchdata_status==1 and Lab_Strength!=0.0 and maturity_status==1:
            Lab_Strength=float(self.lineEdit_strength_input.text())
            A = 0.0
            B = 0.0
            if Lab_Strength < 17.5:
                A = 10
                B = 68
            elif Lab_Strength >= 17.5 and Lab_Strength < 35.0:
                A = 21
                B = 61
            maturity_df_str=maturity_df_g.copy()
            maturity_df_str=maturity_df_str[[mix_design_id,'Time']].dropna()
            maturity_df_str['Mat_Strength']=0
            for i, j in maturity_df_str[mix_design_id].items():
                maturity_df_str['Mat_Strength'][i]=Lab_Strength*(A + B * math.log10(maturity_df_str[mix_design_id][i]*24/1000))/100
            self.lineEdit_calculated_strength.setText(str(round(maturity_df_str['Mat_Strength'].values[-1], 3)))

            global mat_str']

            xtick_location = ax[::48]
            xtick_labels = [x for x in ax[::48]]

            self.MplWidget_2.canvas.axes.clear()
            self.MplWidget_2.canvas.axes.plot(ax, ay,label='Strength-MPa', color='tab:red')
            self.MplWidget_2.canvas.axes.set_ylim(0,)
            #self.MplWidget_2.canvas.axes.legend('Concrete Temperature °C', loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
            self.MplWidget_2.canvas.axes.set_title('Real Time Strength Graph')
            self.MplWidget_2.canvas.axes.set_xlabel('Time Hr')
            self.MplWidget_2.canvas.axes.set_ylabel('Strength MPa')
            self.MplWidget_2.canvas.axes.legend(['Strength-MPa'], loc='lower right')
            self.MplWidget_2.canvas.axes.set_xticks(xtick_location, xtick_labels, rotation=45)
            self.MplWidget_2.canvas.draw()
            self.statusBar().showMessage("Strength Calculated & graphed Successfully!!")
            global strength_status
            strength_status=1

        elif fetchdata_status==0:
           self.statusBar().showMessage("No Data!! Please fetch data first!!")
        elif maturity_status==0:
           self.statusBar().showMessage("Calculate Maturity first!!")
        elif Lab_Strength==0.0:
           self.statusBar().showMessage("Input the Designed Strength!!")
        else:
            self.statusBar().showMessage("Error Please Fetch data, Calculate Maturity and Input Design Strength!!")

    def realtime_temptime_graph(self):
        if fetchdata_status==0:
            self.statusBar().showMessage("No Data!! Please fetch data first!!")
        else:
            ax = sensor_dataframe_g['Time']

            xtick_location = ax[::48]

            self.MplWidget_2.canvas.axes.clear()
            self.MplWidget_2.canvas.axes.plot(ax, ay, color='tab:red',label='Concrete Temperature °C')
            #self.MplWidget_2.canvas.axes.legend('Concrete Temperature °C', loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
            self.MplWidget_2.canvas.axes.set_title('Real Time Temperature Graph')
            self.MplWidget_2.canvas.axes.set_xlabel('Time')
            self.MplWidget_2.canvas.axes.set_ylabel('Temperature °C')
            self.MplWidget_2.canvas.axes.legend(['Sensor Temperature °C'], loc='upper right')
            self.MplWidget_2.canvas.axes.set_xticks(xtick_location, xtick_labels, rotation=45)
            self.MplWidget_2.canvas.draw()
            self.statusBar().showMessage("Data Graphed Successfully!!")


    def Mat_strength_graph(self):
        if fetchdata_status!=0 and maturity_status!=0 and strength_status!=0:
            ax = maturity_df_g[mix_design_id]
            ay = mat_str['Mat_Strength']

            xtick_location = ax[::48]

            self.MplWidget_2.canvas.axes.clear()
            self.MplWidget_2.canvas.axes.plot(ax, ay, color='tab:red', label='Strength Time Graph')
            # self.MplWidget_2.canvas.axes.legend('Concrete Temperature °C', loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
            self.MplWidget_2.canvas.axes.set_title('Strength Maturity Graph')
            self.MplWidget_2.canvas.axes.set_xlabel('Maturity °C-Hr')
            self.MplWidget_2.canvas.axes.set_ylabel('Strength MPa')
            self.MplWidget_2.canvas.axes.legend(['Strength MPa'], loc='lower right')
            self.MplWidget_2.canvas.axes.set_xticks(xtick_location, xtick_labels, rotation=45)
            self.MplWidget_2.canvas.draw()
            self.statusBar().showMessage("Data Graphed Successfully!!")
        elif fetchdata_status==0 or maturity_status==0 or strength_status==0:
            self.statusBar().showMessage("Please fetch data, calculate maturity and strength!!")


    def Str_Train(self):
        global mix_design_id_t
        #Data fetch
        sensor_id = str(self.lineEdit_label_Train_S_input.text())
        mix_design_id_t = str(self.lineEdit_Train_M_input.text())
        datum_temp = float(self.datumtemp_input.text())
        full_id = "/SensorData/lwgbIPQ6lpPSuE9xUhZIqsj3Wsi1/" + sensor_id
        # Initialize the app with a service account, granting admin privileges
        ref = db.reference(full_id).get()
        print('check 1')
        if type(ref) is dict:
            # Create a data frame from dictionery returned by firebase
            sensor_dataframe = pd.DataFrame.from_dict(ref.values())
            sensor_dataframe = sensor_dataframe.apply(pd.to_numeric)
            # Convert the data and time to human readable form
            print(timestamp_array)
            i = 0
            for time in timestamp_array:
                timestamp_array[i] = dt.datetime.fromtimestamp(int(time)).isoformat("#", "minutes")
                i = i + 1
            sensor_dataframe['Time'] = timestamp_array
            sensor_dataframe.drop('timestamp', inplace=True, axis=1)
            print('work')
            global sensor_dataframe_t
            sensor_dataframe_t = sensor_dataframe.copy()
            self.statusBar().showMessage("Data Fetched Successfully!!")
            print('check 2')
            # Maturity calculation
            global sum1
            maturity_df = sensor_dataframe_t.copy()
            series_size = maturity_df['Time'].size
            maturity_df['Time'] = np.arange(0, series_size / 2, 0.5)
            sum1 = 0.0
            print('check 2.1')
            for x, z in maturity_df.items():
                if x == mix_design_id_t:
                    print('check 2.2')
                        print('check 2.3')
                    sum1 = 0.0
            global maturity_df_t
            print('check 3')
            maturity_df_t=maturity_df.copy()
            self.statusBar().showMessage("Maturity calculated Successfully!!")

            mat1 = maturity_df_t[mix_design_id_t][48]
            mat3 = maturity_df_t[mix_design_id_t][144]
            mat7 = maturity_df_t[mix_design_id_t][336]
            mat14 = maturity_df_t[mix_design_id_t][672]
            mat28 = maturity_df_t[mix_design_id_t][1300]
            strength1 = float(self.day1input.text())
            strength3 = float(self.day3input.text())
            strength7 = float(self.day7input.text())
            strength14 = float(self.day14input.text())
            strength28 = float(self.day28input.text())

            data = np.array(
                [[mat1, strength1], [mat3, strength3], [mat7, strength7], [mat14, strength14], [mat28, strength28]])
            mat_strength = pd.DataFrame(data, columns=['maturity', 'strength'])
            print(mat_strength)
            print('check 4')
            # Eliminating NaN or missing input numbers
            mat_strength.fillna(method='ffill', inplace=True)
            global X,y,y_pred
            X = np.array(mat_strength['maturity']).reshape(-1, 1)
            y = np.array(mat_strength['strength']).reshape(-1, 1)
            # Separating the data into independent and dependent variables
            # Converting each dataframe into a numpy array
            # since each dataframe contains only one column
            mat_strength.dropna(inplace=True)
            print('check 5')
            #non linear regression model
            model = Pipeline([ ('poly', PolynomialFeatures(degree=2)),
                ('linear', LinearRegression(fit_intercept=False)) ])
            model.fit(X, y)
            print(model.score(X, y))
            y_pred = model.predict(X)
            self.MplWidget_2.canvas.axes.clear()
            self.MplWidget_2.canvas.axes.scatter(X, y, label ='Actual_Strength', color='b')
            self.MplWidget_2.canvas.axes.plot(X, y_pred, label ='Actual_Strength', color='c',linestyle='dotted')

            self.MplWidget_2.canvas.axes.legend(['Actual_Strength','Trained Curve'], loc='lower right')
    #        self.MplWidget_2.canvas.axes.set_xticks(xtick_location, xtick_labels, rotation=45)
            self.MplWidget_2.canvas.draw()
            self.statusBar().showMessage("Data Graphed Successfully!!")
            global train_model_status
            train_model_status = 1

#Prediction of Data
    def Train_use(self):
        if train_model_status == 0:
            self.statusBar().showMessage("Train Model!!")
            return
        if fetchdata_status==1:
            if maturity_status==0:

            global x1,y_pred1
            print('train5')
            x1 = np.array(maturity_df_g[mix_design_id]).reshape(-1, 1)
            print('train3')
            print(x1)
            y_pred1 = model.predict(x1)
            self.MplWidget_2.canvas.axes.clear()
            self.MplWidget_2.canvas.axes.scatter(X, y, label='Actual_Strength', color='b')
            self.MplWidget_2.canvas.axes.plot(X, y_pred, label='Actual_Strength', color='c',linestyle='dotted')

            self.MplWidget_2.canvas.axes.legend(['Actual_Strength','Trained Curve','Sensor Maturity Strength'], loc='lower right')
            #        self.MplWidget_2.canvas.axes.set_xticks(xtick_location, xtick_labels, rotation=45)
            self.MplWidget_2.canvas.draw()
            self.statusBar().showMessage("Data Graphed Successfully!!")
            global rg_strength_status
            rg_strength_status=1
        elif fetchdata_status==0:
            rg_strength_status = 0
            self.statusBar().showMessage("No Data!! Please fetch data for sensor first!!")

    def weather(self): #load the weather data
        global wdf
        api_key = "afbf9bddb42545a481abfff10eb4bd8a"
        lat = 34.0773
        lon = 72.6210
        api = Api(api_key)
        api.set_granularity('daily')
        forecast = api.get_forecast(lat=lat, lon=lon, hours=239)
        wdict = forecast.get_series(['high_temp', 'low_temp', 'weather'])
        wdf = pd.DataFrame.from_dict(wdict)
        wdf["weather"] = wdf["weather"].astype(str)
        wdf["weather"] = wdf["weather"].str.slice(start=46, stop=75, step=1)
        wdf["Description"] = wdf["weather"].str.slice(start=-40, stop=-2, step=1)
        wdf.drop('datetime', inplace=True, axis=1)
        wdf.drop('weather', inplace=True, axis=1)
        wdf.drop('timestamp_utc', inplace=True, axis=1)
        wdf.set_index('timestamp_local')
        wdf = wdf.reindex(columns=['timestamp_local', 'high_temp', 'low_temp', 'Description'])
        ax2 = wdf['timestamp_local'].astype('str')
        ay20 = wdf['high_temp'].astype('int')
        ay21 = wdf['low_temp'].astype('int')
        #data from the visual crossing
        json_obj = json.load(urllib.request.urlopen(
            "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/34.0773%2C%2072.6210?unitGroup=metric&include=hours&key=ERUMCFV9MKCYQAYQQX8YLECLS&contentType=json"))
        wdf2 = pd.DataFrame()
        for x in json_obj['days']:
            wdf1 = pd.DataFrame.from_dict(x['hours'])
            wdf2 = pd.concat([wdf2, wdf1])
        i = 0

            elif i > 18 & i <= 23:
                night = night + x
                i = i + 1
        day_arr = np.array(day_list)
        night_arr = np.array(night_list)
        wdf['Avg Day Temp'] = pd.Series(day_arr)
        wdf['Avg Night Temp'] = pd.Series(night_arr)
        wdf = wdf.fillna(0)


        ax2 = wdf['timestamp_local'].astype('str')
        ax2 = ax2[0:10]
        ay20 = wdf['high_temp'].astype('int')
        ay20 = ay20[0:10]
        ay21 = wdf['low_temp'].astype('int')
        ay21 = ay21[0:10]
        ay22 = wdf['Avg Day Temp']  # .astype('int')
        ay22 = ay22[0:10]
        ay23 = wdf['Avg Night Temp']  # .astype('int')
        ay23 = ay23[0:10]

        xtick_location = ax2[::1]
        xtick_labels = [x[:10] for x in ax2[::1]]
        self.MplWidget_2.canvas.axes.clear()
        self.MplWidget_2.canvas.axes.plot(ax2,ay20,color='b',label='Highest Day Temperature °C for 10 Days', linewidth=0.35)
        self.MplWidget_2.canvas.axes.plot(ax2,ay21,color='tab:orange',label='Lowest Night Temperature °C for 10 Days', linewidth=0.35)
        self.MplWidget_2.canvas.axes.plot(ax2,ay22,color='tab:purple',label='Average Day Temperature °C for 10 Days', linewidth=0.35,linestyle='dashdot')
        self.MplWidget_2.canvas.axes.plot(ax2,ay23,color='tab:cyan',label='Average Night Temperature °C for 10 Days', linewidth=0.35,linestyle='dashdot')
        self.MplWidget_2.canvas.axes.legend(('Real Time Temperature'), loc='lower right')
        self.MplWidget_2.canvas.axes.set_title('Temperature Time Graph Forecast')
        self.MplWidget_2.canvas.axes.set_xticks(xtick_location, xtick_labels, rotation=45)
        self.MplWidget_2.canvas.draw()
        self.statusBar().showMessage("Weather Forcast Loaded Successfully!!")
        global weather_status
        weather_status =1

    def mat_weather(self): # calculate matuirty and strength for untouched temperature data
        if weather_status==0:
            self.weather()
        if train_model_status == 0:
            self.statusBar().showMessage("Train Model!!")
            return
        if rg_strength_status == 0:
            self.statusBar().showMessage("Calculate Regression Curve for Sensor required for Graph!!")
            return
        print('wm1')
        global wdf_mat
        wdf_mat = pd.DataFrame({'P_Mat': 0}, index=[0])
        print('wm2')
        sum3 = 0.0

        x2 = np.array(wdf_mat["P_Mat"].copy()).reshape(-1, 1)
        print('wm4')

        model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                          ('linear', LinearRegression(fit_intercept=False))])
        model.fit(X, y)
        print(model.score(X, y))

        print('wm5')
        print(model.score(X, y))
        y_pred = model.predict(X) #trained curve
        y_pred1 = model.predict(x1)#current sensor data
        y_pred3 = model.predict(x2) #weather forecast data
        print('wm6')
        self.MplWidget_2.canvas.axes.clear()
        self.MplWidget_2.canvas.axes.scatter(X, y, label='Actual_Strength', color='b')
        self.MplWidget_2.canvas.axes.plot(X, y_pred, label='Trained Curve', color='tab:brown',linestyle='dotted')
        self.MplWidget_2.canvas.axes.plot(x1, y_pred1, label='Sensor_Maturity_Strength', color='tab:orange',linestyle='dashed')
        self.MplWidget_2.canvas.axes.plot(x2, y_pred3, label='Weather_Maturity_Strength', color='k')
        self.MplWidget_2.canvas.axes.set_ylim(0, )
        self.MplWidget_2.canvas.axes.set_xlim(0, )
        # self.MplWidget_2.canvas.axes.legend('Concrete Temperature °C', loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
        self.MplWidget_2.canvas.axes.set_title('Strength Maturity Graph')
        self.MplWidget_2.canvas.axes.set_xlabel('Maturity °C-Hr')
        self.MplWidget_2.canvas.axes.set_ylabel('Strength MPa')
        self.MplWidget_2.canvas.axes.legend(['Actual_Strength','Trained Curve','Sensor_Maturity_Strength','Weather_Maturity_Strength'], loc='lower right')
        #        self.MplWidget_2.canvas.axes.set_xticks(xtick_location, xtick_labels, rotation=45)
        self.MplWidget_2.canvas.draw()
        self.statusBar().showMessage("Data Graphed Successfully!!")
        global mat_weather_status
        mat_weather_status=1

    def mat_weather_ml(self): #training model for temperature data of weather
        if train_model_status == 0:
            self.statusBar().showMessage("Train Model!!")
            return
        if rg_strength_status == 0:
            self.Train_use()
        print('check0')
        mat_pml = sensor_dataframe_t[[mix_design_id_t, 'S_Temp_36_A']].dropna()  # ask for input or use one above
        mat_pml.reset_index(inplace=True)
        mat_pml = mat_pml.rename(columns={'index': 'Time'})
        # seperate out our x and y values
        print('check1.1')
        x_values = np.array(mat_pml['Time'][1:288].copy()).reshape(-1, 1)  # ,'S_Temp_36_A'
        y_values = np.array(mat_pml[mix_design_id_t][1:288].copy()).reshape(-1, 1)
        print(x_values)
        print(y_values)
        # visual
        print('check1')
        # define our polynomial model, with whatever degree we want
        # polynomialFeatures will create a new matrix consisting of all polynomial combinations
        # of the features with a degree less than or equal to the degree we just gave the model (2)
        model = Pipeline([('poly', PolynomialFeatures(degree=5)),
                          ('linear', LinearRegression(fit_intercept=False))])
        model.fit(x_values, y_values)
        print(model.score(x_values, y_values))
        y_pred_ml= model.predict(x_values)
        # transform out polynomial feature
        print('check2')
        print('check2.1')
        print('check3')
        #plot regression
        print('check4')
        wdf4 = [0, 0, 48, 96, 144, 192, 240, 288]
        print('check4.1')
        x3_values = np.array(wdf4).reshape(-1, 1)
        print('check4.2')
        global y_pred_ml1
        y_pred_ml1= model.predict(x3_values)
        print(y_pred_ml1)
        print('check5')
        self.MplWidget_2.canvas.axes.clear()
        self.MplWidget_2.canvas.axes.plot(x_values, y_values) #orignal temp
        self.MplWidget_2.canvas.axes.scatter(x_values, y_pred_ml) #trained
        self.MplWidget_2.canvas.axes.scatter(x3_values, y_pred_ml1) #predicted
        # self.MplWidget_2.canvas.axes.legend('Concrete Temperature °C', loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
        self.MplWidget_2.canvas.axes.set_title('Temperature Time Graph - (ML + Previous Performance)')
        self.MplWidget_2.canvas.axes.set_xlabel('Time 0.5 Hr')
        self.MplWidget_2.canvas.axes.set_ylabel('Temperature °C')
        self.MplWidget_2.canvas.axes.legend(['Temperature °C'], loc='lower right')
        self.MplWidget_2.canvas.draw()
        self.statusBar().showMessage("Model Trained Successfully ML-Temp!!")
        global temp_train_status
        temp_train_status=1


    def mat_weather_ml_str(self):  #strength on combine weather forecaste temp and previous performnace of mix design
        if temp_train_status==0:
            self.statusBar().showMessage("Please train Temperature Model")
            return
        if rg_strength_status == 0:
            self.statusBar().showMessage("Calculate Regression Curve for Sensor required for Graph!!")
            return
        if mat_weather_status==0:
            self.mat_weather()

        print('check6')
        print(wdf_mat)
        print(y_pred_ml1)
        wdf_mat["Temp_P_ML"] = y_pred_ml1
        wdf2_mat = pd.DataFrame({'P_Mat': 0}, index=[0])
        sum3 = 0.0
        print('check6.1')
        for x, z in wdf_mat.items():
            if x == 'Temp_P_ML':
                for j, k in wdf_mat[x].items():
                    sum3 = sum3 + (k - datum_temp)})
                    wdf2_mat = pd.concat([wdf2_mat, new_row.to_frame().T], ignore_index=True)
        x5 = np.array(wdf2_mat["P_Mat"].copy()).reshape(-1, 1)
        print(x5)
        print('check7')
        model = Pipeline([('poly', PolynomialFeatures(degree=3)),
                          ('linear', LinearRegression(fit_intercept=False))])
        model.fit(X, y)
        y_pred_str_ml = model.predict(x5)
        print(y_pred_str_ml)
        print('check8')
        self.MplWidget_2.canvas.axes.clear()
        self.MplWidget_2.canvas.axes.scatter(X, y, label='Actual_Strength', color='b')
        self.MplWidget_2.canvas.axes.plot(X, y_pred, label='Trained Curve', color='tab:brown', linestyle='dotted')
        self.MplWidget_2.canvas.axes.plot(x1, y_pred1, label='Sensor_Maturity_Strength', color='tab:orange', linestyle='dashed')
        self.MplWidget_2.canvas.axes.plot(x5, y_pred_str_ml, label='Weather_ML_Maturity_Strength', color='k')
        self.MplWidget_2.canvas.axes.set_ylim(0, )
        self.MplWidget_2.canvas.axes.set_xlim(0, )
        # self.MplWidget_2.canvas.axes.legend('Concrete Temperature °C', loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
        self.MplWidget_2.canvas.axes.set_title('Strength Maturity Graph')
        self.MplWidget_2.canvas.axes.set_xlabel('Maturity °C-Hr')
        self.MplWidget_2.canvas.axes.set_ylabel('Strength MPa')
        self.MplWidget_2.canvas.axes.legend(
            ['Actual_Strength', 'Trained Curve', 'Sensor_Maturity_Strength', 'Weather_Maturity_Strength'], loc='lower right')
        #self.MplWidget_2.canvas.axes.set_xticks(xtick_location, xtick_labels, rotation=45)
        self.MplWidget_2.canvas.draw()
        self.statusBar().showMessage("Data Graphed Successfully!!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MatplotlibWidget()
    window.show()
    sys.exit(app.exec_())