import mlflow
import mlflow.sklearn
from lightweight_mmm import lightweight_mmm, preprocessing
import pandas as pd
import jax.numpy as jnp
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from utils import evaluate_model_fit

def train_and_register_model(df_path: str):
    df = pd.read_csv(df_path)
    SEED = 105
    mdsp_cols = ["Paid_Views", "Google_Impressions", "Email_Impressions", "Facebook_Impressions", "Affiliate_Impressions"]
    sales_cols = ['Sales']

    data_size = len(df)
    n_media_channels = len(mdsp_cols)
    media_data = df[mdsp_cols].to_numpy()
    target = df['Sales'].to_numpy()
    extra_features = df['is_peak_season'].to_numpy()
    costs = df[mdsp_cols].sum().to_numpy()

    # Split and scale data
    test_data_period_size = 3
    split_point = data_size - test_data_period_size
    media_data_train = media_data[:split_point, ...]
    extra_features_train = extra_features[:split_point, ...]
    extra_features_test = extra_features[split_point:, ...]
    media_data_test = media_data[split_point:, ...]
    target_train = target[:split_point]
    target_test = target[split_point:]

    media_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    target_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    cost_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)

    media_data_train = media_scaler.fit_transform(media_data_train)
    target_train = target_scaler.fit_transform(target_train)
    costs = cost_scaler.fit_transform(costs)

    if extra_features_train.ndim == 1:
        extra_features_test = extra_features_test[:, np.newaxis]
        extra_features_train = extra_features_train[:, np.newaxis]

    # Start an MLflow experiment
    mlflow.set_experiment("Media Mix Model Tuning")

    adstock_models = ["adstock", "hill_adstock", "carryover"]
    degrees_season = [1, 2, 3]

    best_mape = float('inf')
    best_model_info = None
    best_model_run_id = None

    for model_name in adstock_models:
        for degrees in degrees_season:
            with mlflow.start_run():
                mmm = lightweight_mmm.LightweightMMM(model_name=model_name)
                mmm.fit(media=media_data_train,
                        media_prior=costs,
                        target=target_train,
                        extra_features=extra_features_train,
                        number_warmup=1000,
                        number_samples=1000,
                        number_chains=1,
                        degrees_seasonality=degrees,
                        weekday_seasonality=True,
                        seasonality_frequency=365,
                        seed=1)

                # Evaluate model fit on training data
                train_mape, train_r2 = evaluate_model_fit(mmm, target_scaler=target_scaler)

                # Prediction and Evaluation on test data
                prediction = mmm.predict(media=media_data_test,
                                         extra_features=extra_features_test,
                                         target_scaler=target_scaler)
                p = prediction.mean(axis=0)
                test_mape = mean_absolute_percentage_error(target_test, p)
                test_r2 = r2_score(target_test, p)

                # Log parameters and metrics
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("degrees_seasonality", degrees)
                mlflow.log_metric("train_MAPE", train_mape)
                mlflow.log_metric("train_R2", train_r2)
                mlflow.log_metric("test_MAPE", test_mape)
                mlflow.log_metric("test_R2", test_r2)

                # Log model
                mlflow.sklearn.log_model(mmm, "model")

                # Check if this model is the best
                if test_mape < best_mape:
                    best_mape = test_mape
                    best_model_info = (model_name, degrees)
                    best_model_run_id = mlflow.active_run().info.run_id

    # Register the best model
    if best_model_run_id:
        best_model_uri = f"runs:/{best_model_run_id}/model"
        mlflow.register_model(best_model_uri, "MediaMixModel")
    
    return best_model_info, best_model_run_id, mdsp_cols
