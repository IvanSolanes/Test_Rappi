from flask import render_template, request
from flask import redirect, url_for
import os
from PIL import Image
from app.utils import pipeline_model
import pandas as pd
import shap

UPLOAD_FLODER = 'static/uploads'
def base():
    return render_template('base.html')


def index():
    return render_template('index.html')


def faceapp():
    return render_template('faceapp.html')

def getwidth(path):
    img = Image.open(path)
    size = img.size # width and height
    aspect = size[0]/size[1] # width / height
    w = 300 * aspect
    return int(w)

def conclusion():

    if request.method == "POST":
        f1 = request.form['message']
        f = f1.split(';')
        
        lista = []
        for i in f:
            if i == 'False':
                lista.append(False)
            elif i=='True':
                lista.append(True)
            else:
                try:
                    lista.append(float(i))
                except ValueError:
                    lista.append(i)
        data = pd.DataFrame([lista], columns=['genero', 'monto', 'tipo_tc', 'linea_tc', 'interes_tc', 'status_txn',
       'is_prime', 'dcto', 'cashback', 'OS', 'DEVICE_SCORE'])

        # prediction (pass to pipeline model)
        result = pipeline_model(data)
        

        return render_template('conclusion.html',fileupload=True, prediction = result, desc1=lista)


    return render_template('conclusion.html',fileupload=False)