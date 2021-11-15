import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import pickle
from joblib import dump, load
import shap

def pipeline_model(describ):
    num_features = ["monto", "linea_tc", "interes_tc", "dcto"]
    cat_features = ["genero", "tipo_tc","status_txn", "is_prime", "OS", "DEVICE_SCORE"]
    encoder = load('./model/encoder.joblib')
    scaler = load('./model/scaler.joblib')
    X_train_num_scale = scaler.transform(describ[num_features])
    X_train_cat_enco = encoder.transform(describ[cat_features])
    
    processed_data = np.concatenate([X_train_num_scale, X_train_cat_enco], axis=1)
    
    
    fraud_model = load('./model/fraud_model.joblib')
    y_hat = fraud_model.predict(processed_data)
    
    prob_pred = fraud_model.predict_proba(processed_data)[0][1]
    prob_pred = np.round(prob_pred*100,2)
    prob_pred = str(prob_pred) +"%"
    resultado = [y_hat, prob_pred]
    
    columnas_totales = ['monto', 'linea_tc', 'interes_tc', 'dcto', 'genero_--', 'genero_F', 'genero_M', 'tipo_tc_Física', 'tipo_tc_Virtual', 
                        'status_txn_Aceptada', 'status_txn_En proceso', 'status_txn_Rechazada', 'is_prime_False', 'is_prime_True', 'OS_%%', 'OS_.',
                        'OS_ANDROID', 'OS_WEB', 'DEVICE_SCORE_1', 'DEVICE_SCORE_2', 'DEVICE_SCORE_3', 'DEVICE_SCORE_4', 'DEVICE_SCORE_5']
    explainer = shap.TreeExplainer(fraud_model)
    shap_values = explainer.shap_values(processed_data)
    # shap.initjs()
    shap.force_plot(explainer.expected_value[0], shap_values[0], columnas_totales,show=False, matplotlib=True).savefig('./images/shap_pred.png')
    shap.force_plot(explainer.expected_value[0], shap_values[0], columnas_totales,show=False, matplotlib=True).savefig('./static/images/shap_pred.png')
    
    return resultado
    