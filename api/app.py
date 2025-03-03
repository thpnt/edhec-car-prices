from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from preproc.cleaning import clean_data
import joblib
import pandas as pd
import numpy as np
import os

root = os.path.dirname(os.path.dirname(__file__))

# Load the saved preprocessing pipeline and model
preproc_pipeline = joblib.load(f"{root}/models/preproc_pipeline.joblib")
model = joblib.load(f"{root}/models/rf_model.joblib")

app = FastAPI(
    title="Car Price Prediction API",
    description="API for predicting car prices. Use the `/predict` endpoint for single predictions and `/batch_predict` for batch predictions. You can also try out the API interactively using SwaggerUI at `/docs`.",
    version="1.0"
)

class CarFeatures(BaseModel):
    Levy: Optional[float] = Field(default=None, alias="Levy", description="Levy value")
    Manufacturer: Optional[str] = Field(default=None, alias="Manufacturer", description="Car manufacturer")
    Model: Optional[str] = Field(default=None, alias="Model", description="Car model")
    prod_year: Optional[float] = Field(default=None, alias="Prod. year", description="Production year")
    Category: Optional[str] = Field(default=None, alias="Category", description="Car category")
    Leather_interior: Optional[str] = Field(default=None, alias="Leather interior", description="Leather interior flag (Yes/No)")
    Fuel_type: Optional[str] = Field(default=None, alias="Fuel type", description="Type of fuel used")
    Engine_volume: Optional[str] = Field(default=None, alias="Engine volume", description="Engine volume in liters")
    Mileage: Optional[float] = Field(default=None, alias="Mileage", description="Mileage of the car")
    Cylinders: Optional[float] = Field(default=None, alias="Cylinders", description="Number of engine cylinders")
    Gear_box_type: Optional[str] = Field(default=None, alias="Gear box type", description="Type of gearbox")
    Drive_wheels: Optional[str] = Field(default=None, alias="Drive wheels", description="Drive wheels configuration")
    Doors: Optional[str] = Field(default=None, alias="Doors", description="Number or classification of doors")
    Wheel: Optional[str] = Field(default=None, alias="Wheel", description="Wheel side information")
    Color: Optional[str] = Field(default=None, alias="Color", description="Car color")
    Airbags: Optional[float] = Field(default=None, alias="Airbags", description="Number of airbags")
    
    class Config:
        allow_population_by_field_name = True

class MultipleCarFeatures(BaseModel):
    data: List[CarFeatures]

@app.post(
    "/predict",
    tags=["Single Prediction"],
    summary="Predict car price for a single car",
    description="Submit a JSON payload containing car features to obtain a price prediction for a single vehicle."
)
def predict_single(sample: CarFeatures):
    try:
        # Convert the incoming sample to a DataFrame using the aliases for proper matching
        df = pd.DataFrame([sample.dict(by_alias=True)])
        # Replace None values with NaN
        df = df.where(pd.notnull(df), np.nan)
        expected_cols = {"ID"}  # Add any other columns you expect but might be missing
        # handle missing columns for pipeline
        for col in expected_cols:
            if col not in df.columns:
                df[col] = np.nan  # or any dummy value that works with your pipeline
        # Clean the data
        df = clean_data(df)
        # Preprocess the data
        X_processed = preproc_pipeline.transform(df)
        # Predict using the loaded model
        prediction = model.predict(X_processed)
        prediction = np.exp(prediction)  # Convert the log-transformed prediction back to the original scale
        return {"prediction": prediction.tolist()[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post(
    "/batch_predict",
    tags=["Batch Prediction"],
    summary="Predict car prices for multiple cars",
    description="Submit a JSON payload with a list of car feature objects to receive predictions for each sample."
)
def predict_batch(samples: MultipleCarFeatures):
    try:
        # Convert the list of samples to a DataFrame using aliases
        df = pd.DataFrame([s.dict(by_alias=True) for s in samples.data])
        df = df.where(pd.notnull(df), np.nan)
        # Handle missing columns
        expected_cols = {"ID"}  # Add any other columns you expect but might be missing
        for col in expected_cols:
            if col not in df.columns:
                df[col] = np.nan  # or any dummy value that works with your pipeline
        # Clean data
        df = clean_data(df)
        # Preprocess the input data
        X_processed = preproc_pipeline.transform(df)
        # Predict using the loaded model
        predictions = model.predict(X_processed)
        predictions = np.exp(predictions)  # Convert the log-transformed predictions back to the original scale
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
