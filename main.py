import csv
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import xgboost as xgb
import pandas as pd
import math
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SensorPayload(BaseModel):
    solar_v: float = Field(..., description="Raw solar voltage from INA219 #1")
    batt_v: float = Field(..., description="Raw battery voltage from INA219 #2")
    load_i: float = Field(..., description="Raw load current from INA219 #3")
    hour: int = Field(..., ge=0, le=23, description="Hour from DS3231 RTC")
    minute: int = Field(..., ge=0, le=59, description="Minute from DS3231 RTC")


app = FastAPI(title="AI Microgrid Controller API", version="2.0.0")

model = None


@app.on_event("startup")
async def load_xgboost_model():
    global model
    model_path = 'microgrid_xgboost_model.json'
    try:
        if not os.path.exists(model_path):
            logger.error(f"Critical Error: Model file '{model_path}' not found in directory!")
            return
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        logger.info("XGBoost 2-state model loaded successfully on startup.")
    except Exception as e:
        logger.error(f"Failed to load XGBoost model. Error: {e}")


@app.get("/")
async def root():
    return {"message": "Microgrid 2-State AI Server is Running. Access /docs for Swagger UI."}


@app.post("/predict")
async def predict_microgrid_state(data: SensorPayload):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="ML model is not loaded on the server.")

    try:
        time_decimal = data.hour + (data.minute / 60.0)
        hour_sin = math.sin(2 * math.pi * time_decimal / 24.0)
        hour_cos = math.cos(2 * math.pi * time_decimal / 24.0)

        features = pd.DataFrame([{
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'solar_v': data.solar_v,
            'batt_v': data.batt_v,
            'load_i': data.load_i
        }], columns=['hour_sin', 'hour_cos', 'solar_v', 'batt_v', 'load_i'])

        raw_pred = model.predict(features)[0]
        prediction = int(raw_pred)

        action_map = {
            0: "SOLAR_MODE",
            1: "BATTERY_MODE"
        }

        target_action = action_map.get(prediction, "BATTERY_MODE")

        logger.info(f"V_sol:{data.solar_v}V | V_bat:{data.batt_v}V | Load:{data.load_i}A --> State: {target_action}")

        log_file = 'live_microgrid_log.csv'
        file_exists = os.path.isfile(log_file)
        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'solar_v', 'batt_v', 'load_i', 'prediction_class', 'relay_action'])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                data.solar_v,
                data.batt_v,
                data.load_i,
                prediction,
                target_action
            ])

        return {
            "status": "success",
            "prediction_class": prediction,
            "relay_action": target_action
        }

    except Exception as e:
        logger.error(f"Inference pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during ML inference: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)