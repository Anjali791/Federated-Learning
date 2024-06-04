# -*- coding: utf-8 -*-
"""app.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1K3b39QILQQetCF4uNXo-yIf7WIvk1NEd
"""

import flask
import numpy as np
import tensorflow as tf
from flask import Flask, request

app = Flask(__name__)

@app.route("/")
#@app.route("/index")

def index():
	return flask.render_template('index.html')


@app.route("/predict",methods = ['POST'])



def predict():
    if request.method == 'POST':
        a = request.form.get('WMC')
        b = request.form.get('DIT')
        c = request.form.get('NOC')
        d = request.form.get('CBO')
        e = request.form.get('RFC')
        f = request.form.get('LCOM')
        g = request.form.get('CA')
        h = request.form.get('CE')
        i = request.form.get('NPM')
        j = request.form.get('LCOM3')
        k = request.form.get('LOC')
        l = request.form.get('DAM')
        m = request.form.get('MOA')
        n = request.form.get('MFA')
        o = request.form.get('CAM')
        p = request.form.get('IC')
        q = request.form.get('CBM')
        r = request.form.get('AMC')
        s = request.form.get('NR')
        t = request.form.get('NDC')
        u = request.form.get('NML')
        v = request.form.get('NDPV')
        w = request.form.get('MAX(CC)')
        x = request.form.get('AVG(CC)')

        X = np.array([[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x]])
        
        print('datatype is', X.dtype)
        X = X.astype('float64')
        print('datatype is', X.dtype)

        img_rows, img_cols = 1, X.shape[1]
        X1 = X
        X = X1.reshape(X1.shape[0], img_rows, img_cols, 1)

        
        pred = loaded_model.predict(X)
        return flask.render_template('predict.html', response = pred[0][0])


if __name__ == '__main__':
    #json_file = open("model.json","r")
    #loaded_model_json = json_file.read()
    #json_file.close()
    #loaded_model = model_from_json(loaded_model_json)
    #loaded_model.load_weights("model.h5")
    loaded_model = tf.keras.models.load_model('C:/Users/Akhil Bansal/Desktop/Model Deployment (2)/Model Deployment/Model Deployment/xerces1.4.4.1CNNModel.h5', compile=False)
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    app.run(host='0.0.0.0', port=8001, debug=True)