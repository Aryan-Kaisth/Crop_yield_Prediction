from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd

from src.pipelines.prediction_pipeline import PredictionPipeline
from src.logger import logging
from src.exception import CustomException
from app.schemas import CropRequest

app = FastAPI(title="Crop Yield Prediction")

# Load model and preprocessor once at startup
pipeline = PredictionPipeline()

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def build_features(data: CropRequest) -> pd.DataFrame:
    """
    Convert CropRequest to a DataFrame for prediction.
    Only uses original fields (no binned features).
    """
    features = {
        "Region": data.Region,
        "Soil_Type": data.Soil_Type,
        "Crop": data.Crop,
        "Rainfall_mm": data.Rainfall_mm,
        "Temperature_Celsius": data.Temperature_Celsius,
        "Fertilizer_Used": 1 if data.Fertilizer_Used.lower() == "yes" else 0,
        "Irrigation_Used": 1 if data.Irrigation_Used.lower() == "yes" else 0,
        "Weather_Condition": data.Weather_Condition,
        "Days_to_Harvest": data.Days_to_Harvest,
    }
    return pd.DataFrame([features])


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Render the HTML form for user input.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict_crop_yield(data: CropRequest):
    """
    Predict crop yield from user input.
    """
    try:
        logging.info(f"Received prediction request: {data.dict()}")
        df = build_features(data)
        # Make prediction
        pred = pipeline.predict(df)[0]
        logging.info(f"Prediction result: {pred}")
        return {"predicted_yield": float(pred)}
    except CustomException as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Model prediction failed.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error.")
