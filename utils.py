import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def get_target_columns(target_cols, df):
    target_cols_index = [list(df.columns).index(x) for x in target_cols]
    return target_cols_index

def normalize_data(data):
    data_max = data.max(axis=0)
    data_min = data.min(axis=0)
    data = (data - data_min) / (data_max - data_min)
    return data, data_min, data_max

def evaluate_model(model, X_test, Y_test, data_min, data_max,target_cols, pred_len):
    predictions = model.predict(X_test)
    
    # Denormalize predictions and actual values
    Y_test_denorm = Y_test * (data_max[target_cols] - data_min[target_cols]) + data_min[target_cols]
    predictions_denorm = predictions * (data_max[target_cols] - data_min[target_cols]) + data_min[target_cols]
    
    # Initialize metrics storage
    metrics = {
        'mae_per_variable': [],
        'rmse_per_variable': [],
        'mae_per_timestep': [],
        'rmse_per_timestep': []
    }
    
    # Compute metrics for each variable
    for i in range(len(target_cols)):
        mae = mean_absolute_error(Y_test_denorm[:, :, i], predictions_denorm[:, :, i])
        rmse = np.sqrt(mean_squared_error(Y_test_denorm[:, :, i], predictions_denorm[:, :, i]))
        metrics['mae_per_variable'].append(mae)
        metrics['rmse_per_variable'].append(rmse)
        print(f'R{target_cols[i]}: MAE = {mae:.4f}, RMSE = {rmse:.4f}')
    
    # Compute metrics for each prediction timestep
    for t in range(pred_len):
        mae = mean_absolute_error(Y_test_denorm[:, t, :], predictions_denorm[:, t, :])
        rmse = np.sqrt(mean_squared_error(Y_test_denorm[:, t, :], predictions_denorm[:, t, :]))
        metrics['mae_per_timestep'].append(mae)
        metrics['rmse_per_timestep'].append(rmse)
        print(f'Timestep {t+1}: MAE = {mae:.4f}, RMSE = {rmse:.4f}')
    
    # Overall metrics
    metrics['overall_mae'] = np.mean(metrics['mae_per_variable'])
    metrics['overall_rmse'] = np.mean(metrics['rmse_per_variable'])
    print(f'\nOverall MAE: {metrics["overall_mae"]:.4f}')
    print(f'Overall RMSE: {metrics["overall_rmse"]:.4f}')
    
    return metrics, Y_test_denorm, predictions_denorm

def plot_predictions(predictions_denorm, y_test_denorm, target_cols):
    # plt.figure(figsize=(12,5))
    # plt.plot(predictions_denorm[-500:,0,0],'--', label='prediction')
    # plt.plot(y_test_denorm[-500:,0,0], label='actual')
    # plt.legend()
    #var_list = ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    plt.figure(figsize=(12,4*len(target_cols)))
    for i in range(len(target_cols)):
        plt.subplot(len(target_cols),1,i+1)
        plt.title(target_cols[i])
        #plt.ylabel('Power Consumption (Watt hour)')
        plt.plot(y_test_denorm[-500:,0,i], '-', color='green', label='actual')
        steps = predictions_denorm.shape[1]-1
        for j in range(0,predictions_denorm.shape[1],steps if steps > 1 else 1):
            plt.plot(predictions_denorm[-500:,j,i],'--', label=f'prediction step{j+1}')
    
        plt.legend()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    # Encode plot to base64 for HTML
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    return plot_data

def create_sequences(data, seq_len, pred_len, target_cols):
    xs, ys = [], []
    y_data = data[:,target_cols]
    for i in range(len(data) - seq_len - pred_len):
        # Ensure no future data in input sequence
        assert i + seq_len <= len(data), "Input sequence contains future data"
        # Ensure target sequence follows input sequence
        assert i + seq_len + pred_len <= len(data), "Target sequence out of bounds"
        x = data[i:i+seq_len, :]
        y = y_data[i+seq_len:i+seq_len+pred_len, :]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def preprocessing(dataset):
    file_path = f'./{dataset}_df_2.csv'
    if dataset == 'traffic':
        traffic_df = pd.read_csv(file_path)
        traffic_df = traffic_df[::10].reset_index(drop=True)
        traffic_df.drop(columns=['Time'],inplace=True)
        data = np.array(traffic_df)
        target_cols = ['R1','R2','R3','R4','R5','R6']
        target_cols_index = get_target_columns(target_cols, traffic_df)
    elif dataset=='air_quality':
        air_quality_df = pd.read_csv(file_path)
        air_quality_df.drop(columns=['No', 'year', 'month','day','hour','RAIN','wd','station'], inplace=True)
        air_quality_df = air_quality_df.ffill()
        data = np.array(air_quality_df)
        target_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
        target_cols_index = get_target_columns(target_cols, air_quality_df)
    elif dataset=='epc':
        epc_df = pd.read_csv(file_path)
        epc_df.replace('?',np.nan,inplace=True)
        epc_df = epc_df.ffill()
        epc_df = epc_df.drop(columns=['Date','Time'])
        epc_df = epc_df.astype('float')
        data = np.array(epc_df)
        target_cols = ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
        target_cols_index = get_target_columns(target_cols, epc_df)
    elif dataset=='temp':
        temp_df = pd.read_csv(file_path)
        temp_df.drop(columns=
                     ['1:Date','2:Time','12:Precipitacion','19:Exterior_Entalpic_1',
                       '20:Exterior_Entalpic_2', '21:Exterior_Entalpic_turbo','24:Day_Of_Week'], inplace=True)
        # temp_df.drop(columns=['#','Unnamed: 1','1:Date','2:Time','12:Precipitacion','19:Exterior_Entalpic_1',
        #                '20:Exterior_Entalpic_2', '21:Exterior_Entalpic_turbo','24:Day_Of_Week'], inplace=True)
        #temp_df.fillna(20, inplace=True)
        data = np.array(temp_df)
        target_cols = ['3:Temperature_Comedor_Sensor', '4:Temperature_Habitacion_Sensor']
        target_cols_index = get_target_columns(target_cols, temp_df)

    elif dataset=='stock':
        stock_df = pd.read_csv(file_path)
        stock_df.drop(columns=['Date'], inplace=True)
        data = np.array(stock_df)
        target_cols = ['Open','Close']
        target_cols_index = get_target_columns(target_cols, stock_df)

    return data, target_cols_index, target_cols