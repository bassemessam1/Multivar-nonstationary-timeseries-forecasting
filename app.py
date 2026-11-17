from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import os
from  fusion_model import *
from mhsa_model import *
from utils import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

# Model configuration options
MODEL_OPTIONS = {
    "prediction_types": ["single", "multi"],
    "architectures": ["mhsa", "fusion"],
    "datasets": ["traffic", "air_quality", "epc", "temp", "stock"],
    "window_sizes": {
        "traffic": [3, 6, 12, 24],
        "air_quality": [3, 6, 12, 24],
        "epc": [5, 10, 20, 30],
        "temp": [4, 8, 12, 16],
        "stock": [5, 10, 20, 40]
    },
    "horizon_sizes": {
        "traffic": [3, 6, 12, 24],
        "air_quality": [3, 6, 12, 24],
        "epc": [5, 10, 20, 40],
        "temp": [4, 8, 12, 16],
        "stock": [5, 10, 20, 40]
    }
}

@app.route('/')
def index():
    return render_template('index.html', options=MODEL_OPTIONS)

@app.route('/get_options', methods=['POST'])
def get_options():
    data = request.json
    response = {
        "window_options": MODEL_OPTIONS["window_sizes"][data["dataset"]],
        "horizon_options": MODEL_OPTIONS["horizon_sizes"][data["dataset"]],
        "fixed_horizon": 1 if data["prediction_type"] == "single" else None,
        "fixed_window": max(MODEL_OPTIONS["window_sizes"][data["dataset"]]) if data["prediction_type"] == "multi" else None
    }
    return jsonify(response)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        selections = {
            'prediction_type': request.form['prediction_type'],
            'architecture': request.form['architecture'],
            'dataset': request.form['dataset'],
            'window': request.form.get('window', str(max(MODEL_OPTIONS["window_sizes"][request.form['dataset']]))),
            'horizon': request.form.get('horizon', '1')
        }
        
        model_name = f"model_{selections['architecture']}_{selections['prediction_type']}_{selections['dataset']}_w{selections['window']}_h{selections['horizon']}.keras"
        data, target_cols_index, target_cols = preprocessing(dataset=selections['dataset'])
        data, data_min, data_max = normalize_data(data)
        if selections['architecture']=='fusion':
            print(f'fusion selected - ./models/{model_name}')
            loaded_model = tf.keras.models.load_model(f'./models/{model_name}', compile=False,
                                            custom_objects={
                                                    'LSTMForecastingModel': LSTMForecastingModel,
                                                    'EnhancedFCM': EnhancedFCM,
                                                    'MHSA': MHSA,
                                                    'CausalAttention': CausalAttention,
                                                    'SoftAttention': SoftAttention,
                                                    'AdaptiveAttentionFusion': AdaptiveAttentionFusion,
                                                })
        if selections['architecture']=='mhsa':
            loaded_model = tf.keras.models.load_model(f'./models/{model_name}', compile=False,
                                            custom_objects={
                                                    'LSTMForecastingModel': LSTMForecastingModel,
                                                    'EnhancedFCM': EnhancedFCM,
                                                    'MHSA': MHSA,
                                                })
            
        X,y = create_sequences(data=data, seq_len=int(selections['window']), pred_len=int(selections['horizon']), target_cols=target_cols_index)
        metrics, Y_test_denorm, predictions_denorm = evaluate_model(model=loaded_model, X_test=X, Y_test= y , data_min=data_min, 
                                                                    data_max=data_max,target_cols=target_cols_index, pred_len=1)
        plot_data = plot_predictions(predictions_denorm, Y_test_denorm, target_cols)
        
        results = {
            'model_used': model_name,
            'input_shape': str(X.shape),
            'prediction_shape': str(predictions_denorm.shape),
            'plot': plot_data,
            'metrics': metrics,
            'parameters': {
                'prediction_type': selections['prediction_type'],
                'architecture': selections['architecture'],
                'dataset': selections['dataset'],
                'window': selections['window'],
                'horizon': selections['horizon']
            }
        }
        loaded_model = None
        
        return render_template('predict.html', 
                            results=results)
    
    except Exception as e:
        return render_template('index.html', 
                             error=str(e),
                             options=MODEL_OPTIONS)

if __name__ == '__main__':
    app.run(debug=True)