# Multivariate Non-Stationary Time-Series Forecasting

A Flask web application for experimenting with multivariate time-series forecasting models. The application supports single-step and multi-step prediction using the MHSA and Fusion architectures, and reports predictions together with per-variable and overall MAE/RMSE metrics.

## Features

- Web-based model and dataset selection.
- Single-step and multi-step forecasting.
- MHSA and Fusion model architectures.
- Forecasting for traffic, air quality, household electricity, temperature, and stock datasets.
- Prediction visualizations rendered in the browser.
- MAE and RMSE metrics per variable, per timestep, and overall.

## Project structure

```text
.
├── app.py              # Flask application and HTTP routes
├── fusion_model.py     # Fusion forecasting architecture
├── mhsa_model.py       # MHSA forecasting architecture
├── utils.py            # Data preparation, evaluation, and plotting helpers
├── datasets/           # CSV datasets
├── models/             # Saved Keras models
├── templates/          # Flask HTML templates
├── static/             # Web assets
└── requirements.txt    # Python dependencies
```

## Requirements

- Python 3.10 or newer
- TensorFlow
- Flask
- NumPy, pandas, scikit-learn, and Matplotlib

Install the dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate       # Windows PowerShell
pip install -r requirements.txt
```

## Data setup

The application currently constructs dataset paths as `./datasets/{dataset}_df_2.csv`.

The supported dataset names and expected files are:

| Dataset | Expected file | Forecast targets |
| --- | --- | --- |
| `traffic` | `traffic_df_2.csv` | `R1`–`R6` |
| `air_quality` | `air_quality_df_2.csv` | `PM2.5`, `PM10`, `SO2`, `NO2`, `CO`, `O3` |
| `epc` | `epc_df_2.csv` | `Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3` |
| `temp` | `temp_df_2.csv` | `3:Temperature_Comedor_Sensor`, `4:Temperature_Habitacion_Sensor` |
| `stock` | `stock_df_2.csv` | `Open`, `Close` |

## Running the application

From the repository root:

```bash
source .venv/bin/activate
python app.py
```

Open <http://127.0.0.1:5000> in a browser, then choose:

1. A prediction type (`single` or `multi`).
2. A model architecture (`mhsa` or `fusion`).
3. A dataset.
4. The available input window and forecast horizon.

For multi-step forecasting, the interface uses the largest configured input window for the selected dataset. For single-step forecasting, the horizon is fixed at `1`.

## Model files

Model files must follow this naming convention:

```text
model_{architecture}_{prediction_type}_{dataset}_w{window}_h{horizon}.keras
```

For example:

```text
models/model_fusion_multi_air_quality_w24_h3.keras
```

Only the example air-quality Fusion model is currently included in the repository. Other combinations require compatible trained `.keras` files to be added to `models/`.

## Evaluation

When a request is submitted, the application:

1. Loads and preprocesses the selected CSV.
2. Normalizes the data using column-wise min/max scaling.
3. Builds input and target sequences.
4. Loads the selected Keras model.
5. Denormalizes predictions and calculates MAE/RMSE.
6. Displays metrics, tensor shapes, and prediction plots.

## Notes

- The Flask server runs with `debug=True` when started through `app.py`; disable debug mode before deploying publicly.
- The available window and horizon values are configured in `MODEL_OPTIONS` in `app.py`.
- Saved models must be compatible with the custom layers and classes defined in `fusion_model.py` and `mhsa_model.py`.
