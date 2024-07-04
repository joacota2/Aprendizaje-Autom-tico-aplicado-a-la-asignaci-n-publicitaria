import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from lightweight_mmm import lightweight_mmm

def evaluate_model_fit(media_mix_model, target_scaler=None):
    if not hasattr(media_mix_model, "trace"):
        raise lightweight_mmm.NotFittedModelError("Model needs to be fit first before attempting to plot its fit.")
    
    target_train = media_mix_model._target
    posterior_pred = media_mix_model.trace["mu"]
    if target_scaler:
        posterior_pred = target_scaler.inverse_transform(posterior_pred)
        target_train = target_scaler.inverse_transform(target_train)
    
    mape = mean_absolute_percentage_error(target_train, posterior_pred.mean(axis=0))
    r2 = r2_score(target_train, posterior_pred.mean(axis=0))
    
    return mape, r2
