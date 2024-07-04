from lightweight_mmm import lightweight_mmm, preprocessing
import pandas as pd
import jax.numpy as jnp
import time
import numpy as np

def estimate_training_time(df_path: str, sample_size: int = 10) -> float:
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
    split_point = data_size - sample_size
    media_data_train = media_data[:split_point, ...]
    target_train = target[:split_point]
    extra_features_train = extra_features[:split_point, ...]

    media_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    target_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    cost_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)

    media_data_train = media_scaler.fit_transform(media_data_train)
    target_train = target_scaler.fit_transform(target_train)
    costs = cost_scaler.fit_transform(costs)

    if extra_features_train.ndim == 1:
        extra_features_train = extra_features_train[:, np.newaxis]

    adstock_models = ["adstock", "hill_adstock", "carryover"]
    degrees_season = [1, 2, 3]

    # Measure time for a small portion of the training data
    sample_media_data_train = media_data_train[:sample_size]
    sample_target_train = target_train[:sample_size]
    sample_extra_features_train = extra_features_train[:sample_size]

    start_time = time.time()
    for model_name in adstock_models:
        for degrees in degrees_season:
            mmm = lightweight_mmm.LightweightMMM(model_name=model_name)
            mmm.fit(media=sample_media_data_train,
                    media_prior=costs,
                    target=sample_target_train,
                    extra_features=sample_extra_features_train,
                    number_warmup=1000,
                    number_samples=1000,
                    number_chains=1,
                    degrees_seasonality=degrees,
                    weekday_seasonality=True,
                    seasonality_frequency=365,
                    seed=1)
    end_time = time.time()
    sample_training_time = end_time - start_time

    # Extrapolate total training time
    total_estimated_time = (sample_training_time / sample_size) * len(media_data_train)
    
    return total_estimated_time
