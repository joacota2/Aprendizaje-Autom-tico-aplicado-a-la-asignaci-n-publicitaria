from zenml.pipelines import pipeline
from zenml.steps import step, Output
import pandas as pd
import jax.numpy as jnp
import numpy as np
from typing import Tuple, List
from lightweight_mmm import lightweight_mmm
from lightweight_mmm import preprocessing
from sklearn.metrics import mean_absolute_percentage_error
import joblib

# Step 1: Data Loading
@step
def load_data() -> pd.DataFrame:
    df = pd.read_csv('MMM.csv')
    return df

# Step 2: Preprocessing
@step
def preprocess_data(df: pd.DataFrame, test_data_period_size: int) -> Output(
    media_data_train=np.ndarray,
    target_train=np.ndarray,
    media_data_test=np.ndarray,
    target_test=np.ndarray,
    extra_features_train=np.ndarray,
    extra_features_test=np.ndarray,
    costs=np.ndarray,
    mdsp_cols=list,
    target_scaler=preprocessing.CustomScaler
):
    mdsp_cols = ["Paid_Views", "Google_Impressions", "Email_Impressions", "Facebook_Impressions", "Affiliate_Impressions"]

    data_size = len(df)
    media_data = df[mdsp_cols].to_numpy()
    target = df['Sales'].to_numpy()
    extra_features = df['is_peak_season'].to_numpy().reshape(-1, 1)  # Ensure extra_features is 2D
    costs = df[mdsp_cols].sum().to_numpy()

    # Split and scale data
    split_point = data_size - test_data_period_size
    media_data_train = media_data[:split_point, ...]
    media_data_test = media_data[split_point:, ...]
    extra_features_train = extra_features[:split_point, ...]
    extra_features_test = extra_features[split_point:, ...]
    target_train = target[:split_point]
    target_test = target[split_point:]

    media_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    target_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    cost_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)

    media_data_train = media_scaler.fit_transform(media_data_train)
    media_data_test = media_scaler.transform(media_data_test)
    target_train = target_scaler.fit_transform(target_train)
    target_test = target_scaler.transform(target_test)
    costs = cost_scaler.fit_transform(costs)

    # Convert JAX arrays to numpy arrays
    media_data_train = np.array(media_data_train)
    media_data_test = np.array(media_data_test)
    target_train = np.array(target_train)
    target_test = np.array(target_test)
    extra_features_train = np.array(extra_features_train)
    extra_features_test = np.array(extra_features_test)
    costs = np.array(costs)

    return media_data_train, target_train, media_data_test, target_test, extra_features_train, extra_features_test, costs, mdsp_cols, target_scaler

# Step 3: Hyperparameter Tuning and Model Selection
@step
def hyperparameter_tuning(
    media_data_train: np.ndarray,
    target_train: np.ndarray,
    media_data_test: np.ndarray,
    target_test: np.ndarray,
    extra_features_train: np.ndarray,
    extra_features_test: np.ndarray,
    costs: np.ndarray,
    mdsp_cols: List[str],
    target_scaler: preprocessing.CustomScaler
) -> Output(best_model_name=str, best_degrees=int, best_mape=float):
    adstock_models = ["adstock", "hill_adstock", "carryover"]
    degrees_season = [1, 2, 3]
    best_mape = float("inf")
    best_model_name = None
    best_degrees = None

    for model_name in adstock_models:
        for degrees in degrees_season:
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
            
            prediction = mmm.predict(
                media=media_data_test,
                extra_features=extra_features_test,
                target_scaler=target_scaler)
            p = prediction.mean(axis=0)

            mape = mean_absolute_percentage_error(target_test, p)
            print(f"model_name={model_name} degrees={degrees} MAPE={mape} samples={p[:3]}")
            
            if mape < best_mape:
                best_mape = mape
                best_model_name = model_name
                best_degrees = degrees
    
    return best_model_name, best_degrees, float(best_mape)

# Step 4: Train Final Model
@step
def train_final_model(
    best_model_name: str,
    best_degrees: int,
    media_data_train: np.ndarray,
    target_train: np.ndarray,
    extra_features_train: np.ndarray,
    costs: np.ndarray,
    mdsp_cols: List[str]
) -> lightweight_mmm.LightweightMMM:
    mmm = lightweight_mmm.LightweightMMM(model_name=best_model_name)
    number_warmup = 1000
    number_samples = 1000
    SEED = 105

    mmm.fit(
        media=media_data_train,
        degrees_seasonality=best_degrees,
        media_prior=costs,
        extra_features=extra_features_train,
        target=target_train,
        number_warmup=number_warmup,
        number_samples=number_samples,
        media_names=mdsp_cols,
        seed=SEED)
    
    return mmm

# Custom save and load functions
def save_lightweight_mmm(model, model_path):
    model_params = {
        'model_name': model.model_name,
        'degrees_seasonality': model.degrees_seasonality,
        'media_names': model.media_names,
        'fit_result': model.fit_result,
        'media': model.media,
        'target': model.target,
        'extra_features': model.extra_features,
        'media_prior': model.media_prior,
        'weekday_seasonality': model.weekday_seasonality,
        'seasonality_frequency': model.seasonality_frequency
    }
    joblib.dump(model_params, model_path)

def load_lightweight_mmm(model_path):
    model_params = joblib.load(model_path)
    model = lightweight_mmm.LightweightMMM(model_name=model_params['model_name'])
    model.degrees_seasonality = model_params['degrees_seasonality']
    model.media_names = model_params['media_names']
    model.fit_result = model_params['fit_result']
    model.media = model_params['media']
    model.target = model_params['target']
    model.extra_features = model_params['extra_features']
    model.media_prior = model_params['media_prior']
    model.weekday_seasonality = model_params['weekday_seasonality']
    model.seasonality_frequency = model_params['seasonality_frequency']
    return model

# Step 5: Save Model
@step
def save_model(model: lightweight_mmm.LightweightMMM, model_path: str = "final_model.pkl") -> str:
    save_lightweight_mmm(model, model_path)
    return model_path

# Pipeline Definition
@pipeline
def mmm_pipeline_hyper(
    load_data_step,
    preprocess_step,
    hyperparameter_tuning_step,
    train_final_model_step,
    save_model_step
):
    df = load_data_step()
    media_data_train, target_train, media_data_test, target_test, extra_features_train, extra_features_test, costs, mdsp_cols, target_scaler = preprocess_step(df, test_data_period_size=8)
    best_model_name, best_degrees, best_mape = hyperparameter_tuning_step(media_data_train, target_train, media_data_test, target_test, extra_features_train, extra_features_test, costs, mdsp_cols, target_scaler)
    final_model = train_final_model_step(best_model_name, best_degrees, media_data_train, target_train, extra_features_train, costs, mdsp_cols)
    model_path = save_model_step(final_model)

# Pipeline Execution
def run():
    mmm_p = mmm_pipeline_hyper(
        load_data_step=load_data(),
        preprocess_step=preprocess_data(),
        hyperparameter_tuning_step=hyperparameter_tuning(),
        train_final_model_step=train_final_model(),
        save_model_step=save_model()
    )
    mmm_p.run()

if __name__ == "__main__":
    run()
