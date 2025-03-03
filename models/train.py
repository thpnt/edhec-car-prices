import pandas as pd
import numpy as np
import os
import joblib
root = os.path.dirname(os.path.dirname(__file__))
from preproc.preprocessing import create_full_pipeline
from preproc.cleaning import clean_data
from preproc.columns import CAT_COLS,NUM_COLS,ORDINAL_FEATURES,ONE_HOT_FEATURES,SCALED_FEATURES,ALL_NUMERIC,OTHER_NUM_FEATURES
from sklearn.ensemble import RandomForestRegressor


if __name__ == "__main__":
    # Load the training data
    data = pd.read_csv(os.path.join(root, "data", "train.csv"))


    # Clean the data
    data = clean_data(data)

    # Preprocess the data for training
    X_train = data.drop(columns="Price")
    y_train = np.log(data["Price"])
    preproc_pipeline = create_full_pipeline(X_train, ONE_HOT_FEATURES, ORDINAL_FEATURES, SCALED_FEATURES)
    X_train = preproc_pipeline.transform(X_train)
    print("Transformed training data shape:", X_train.shape)

    # Save the preprocessing pipeline
    joblib.dump(preproc_pipeline, os.path.join(root, "models", "preproc_pipeline.joblib"))

    # Save the preprocessed training data
    X_train.to_csv(os.path.join(root, "data", "X_train_preprocessed.csv"), index=False)
    y_train.to_csv(os.path.join(root, "data", "y_train_preprocessed.csv"), index=False)
    print("Preprocessed training data saved.")

    # =============================================================================

    # Define the model
    rf_model = RandomForestRegressor(max_depth=None, min_samples_split=2, n_estimators=250, random_state=42)

    # Train the model
    rf_model.fit(X_train, y_train)
    print("Model trained.")

    # Save the model
    joblib.dump(rf_model, os.path.join(root, "models", "rf_model.joblib"), compress=3)
    print("Model saved.")
    
    # Print training RMSE
    from sklearn.metrics import mean_squared_error
    y_train_pred = rf_model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    print("Training RMSE:", train_rmse)


