from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd
import io
import matplotlib.pyplot as plt
from lightweight_mmm import plot
from train_model import train_and_register_model
from predict_model import make_prediction, MediaData
from estimate_training_time import estimate_training_time
import os
import threading
import mlflow
import mlflow.pyfunc
from lightweight_mmm import lightweight_mmm

app = FastAPI()

model_info = {"model": None, "media_scaler": None, "target_scaler": None, "mdsp_cols": None}  # Initialize global model info

def async_train_model(df_path: str):
    best_model_info, best_model_run_id, mdsp_cols = train_and_register_model(df_path)
    model_info["model"] = best_model_info
    model_info["run_id"] = best_model_run_id
    model_info["mdsp_cols"] = mdsp_cols

def load_model_from_mlflow(model_name: str, model_version: str):
    model_uri = f"models:/{model_name}/{model_version}"
    return mlflow.pyfunc.load_model(model_uri)

@app.on_event("startup")
def load_model():
    try:
        # Replace "MediaMixModel" and "1" with your model name and version
        model_name = "MediaMixModel"
        model_version = "1"
        model_uri = f"models:/{model_name}/{model_version}"
        loaded_model = lightweight_mmm.LightweightMMM.load_model(model_uri)
        
        model_info["model"] = loaded_model
        model_info["mdsp_cols"] = ["Paid_Views", "Google_Impressions", "Email_Impressions", "Facebook_Impressions", "Affiliate_Impressions"]
        print("Model loaded successfully from MLflow.")
    except Exception as e:
        print(f"Failed to load model from MLflow: {str(e)}")

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    try:
        # Ensure the directory exists
        os.makedirs("../data", exist_ok=True)
        
        df_path = f"../data/{file.filename}"
        with open(df_path, "wb") as f:
            f.write(file.file.read())

        # Estimate training time
        estimated_training_time = estimate_training_time(df_path)

        # Start training in a separate thread
        threading.Thread(target=async_train_model, args=(df_path,)).start()
        
        return {"detail": "Training job started", "estimated_training_time_seconds": estimated_training_time}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

@app.post("/predict/")
def predict(media_data: MediaData):
    try:
        prediction = make_prediction(media_data)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/plot/media_channel_posteriors")
def plot_media_channel_posteriors():
    try:
        # Logic to load model and plot
        if model_info["model"] is None:
            raise HTTPException(status_code=400, detail="Model not trained yet. Upload a CSV to train the model.")

        fig = plot.plot_media_channel_posteriors(media_mix_model=model_info["model"], channel_names=model_info["mdsp_cols"])
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plotting failed: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
