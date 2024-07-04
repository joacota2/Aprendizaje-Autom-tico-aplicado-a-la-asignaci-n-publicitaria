from zenml.pipelines import pipeline
from zenml.steps import step
import pandas as pd
import jax.numpy as jnp
import numpy as np
from typing import Tuple
from lightweight_mmm import lightweight_mmm
from lightweight_mmm import preprocessing

# Step 1: Data Loading
@step
def load_data() -> pd.DataFrame:
    df = pd.read_csv('MMM.csv')
    return df

# Step 2: Preprocessing
@step
def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    mdsp_cols = ["Paid_Views", "Google_Impressions", "Email_Impressions", "Facebook_Impressions", "Affiliate_Impressions"]

    data_size = len(df)
    media_data = df[mdsp_cols].to_numpy()
    target = df['Sales'].to_numpy()
    extra_features = df['is_peak_season'].to_numpy().reshape(-1, 1)  # Ensure extra_features is 2D
    costs = df[mdsp_cols].sum().to_numpy()

    # Split and scale data
    test_data_period_size = 0
    split_point = data_size - test_data_period_size
    media_data_train = media_data[:split_point, ...]
    extra_features_train = extra_features[:split_point, ...]
    target_train = target[:split_point]
    
    media_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    extra_features_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    target_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    cost_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)

    media_data_train = media_scaler.fit_transform(media_data_train)
    target_train = target_scaler.fit_transform(target_train)
    costs = cost_scaler.fit_transform(costs)

    # Convert JAX arrays to numpy arrays
    media_data_train = np.array(media_data_train)
    target_train = np.array(target_train)
    extra_features_train = np.array(extra_features_train)
    costs = np.array(costs)

    return media_data_train, target_train, extra_features_train, costs, mdsp_cols

# Step 3: Model Training
@step
def train_model(media_data_train: np.ndarray, target_train: np.ndarray, extra_features_train: np.ndarray, costs: np.ndarray, mdsp_cols: list):
    mmm = lightweight_mmm.LightweightMMM(model_name="hill_adstock")
    number_warmup=1000
    number_samples=1000
    SEED = 105

    mmm.fit(
        media=media_data_train,
        degrees_seasonality=3,
        media_prior=costs,
        extra_features=extra_features_train,
        target=target_train,
        number_warmup=number_warmup,
        number_samples=number_samples,
        media_names=mdsp_cols,
        seed=SEED)
    return mmm

# Pipeline Definition
@pipeline
def mmm_pipeline(load_data_step, preprocess_step, train_model_step):
    df = load_data_step()
    media_data_train, target_train, extra_features_train, costs, mdsp_cols = preprocess_step(df)
    model = train_model_step(media_data_train, target_train, extra_features_train, costs, mdsp_cols)

# Pipeline Execution
def run():
    mmm_p = mmm_pipeline(
        load_data_step=load_data(),
        preprocess_step=preprocess_data(),
        train_model_step=train_model()
    )
    mmm_p.run()

if __name__ == "__main__":
    run()
