import pandas as pd
import numpy as np

def clean_data(df):
    """
    Clean the data by fixing data types and creating useful columns.
    """
    # Remove obvious outliers (if applicable)
    if "Price" in df.columns:
        df = df[df.Price < 900000]
    
    # Convert "Levy" to float
    df.Levy = df.Levy.replace('-', np.nan)
    df.Levy = df.Levy.astype(float)
    
    # Convert "Mileage" to float only if it's a string, otherwise leave it as is
    # Mileage is a string during training but float at inference time
    if df.Mileage.dtype == object:
        df.Mileage = df.Mileage.str.replace(' km', '').str.replace(',', '').astype(float)
    
    # Create a "turbo" column from "Engine volume" if it is a string
    if df["Engine volume"].dtype == object:
        df["turbo"] = df["Engine volume"].str.contains("Turbo", case=False)
        df["turbo"] = df["turbo"].astype(int)
        # Remove "Turbo" from "Engine volume" and convert to float
        df["Engine volume"] = df["Engine volume"].str.replace(" Turbo", "").astype(float)
    else:
        # If Engine volume is already numeric, we assume turbo is not embedded in it.
        # You can decide how to handle this scenario.
        df["turbo"] = 0  # or any default value
        
    return df
