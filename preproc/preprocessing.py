import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# =============================================================================
# Helper Functions and Custom Transformers
# =============================================================================

def compute_vif(df):
    """Compute VIF for each feature in the DataFrame."""
    df_const = add_constant(df)
    vif_data = pd.DataFrame()
    vif_data["feature"] = df_const.columns
    vif_data["VIF"] = [variance_inflation_factor(df_const.values, i)
                       for i in range(df_const.shape[1])]
    return vif_data[vif_data["feature"] != "const"]

class DataFrameConverter(BaseEstimator, TransformerMixin):
    """Convert a NumPy array back into a DataFrame using provided column names."""
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return pd.DataFrame(X, columns=self.columns)

class InteractionCreator(BaseEstimator, TransformerMixin):
    """Create interaction features."""
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = X.copy()
        if "Cylinders" in X.columns and "Fuel type" in X.columns:
            X["Fuel type x Cylinders"] = X["Cylinders"] * X["Fuel type"]
        if "Cylinders" in X.columns and "Engine volume" in X.columns:
            X["Engine volume x Cylinders"] = X["Cylinders"] * X["Engine volume"]
        if "Mileage" in X.columns and "Prod. year" in X.columns:
            X["Mileage x Prod. year"] = X["Mileage"] * X["Prod. year"]
        return X

class VIFReducer(BaseEstimator, TransformerMixin):
    """Iteratively remove features with VIF above a threshold."""
    def __init__(self, thresh=10.0):
        self.thresh = thresh
        self.selected_features_ = None
    def fit(self, X, y=None):
        df_reduced = X.copy()
        iteration = 1
        while True:
            vif_df = compute_vif(df_reduced)
            max_vif = vif_df["VIF"].max()
            if max_vif > self.thresh:
                feature_to_drop = vif_df.sort_values("VIF", ascending=False).iloc[0]["feature"]
                print(f"Iteration {iteration}: Dropping '{feature_to_drop}' (VIF = {max_vif:.2f})")
                df_reduced = df_reduced.drop(columns=[feature_to_drop])
                iteration += 1
            else:
                break
        self.selected_features_ = df_reduced.columns.tolist()
        print("Final selected features based on VIF:")
        print(self.selected_features_)
        return self
    def transform(self, X, y=None):
        return X[self.selected_features_]

def drop_turbo_column(X):
    """Drop the 'turbo' column if present."""
    if "turbo" in X.columns:
        return X.drop(columns="turbo")
    return X

# =============================================================================
# First Pipeline: Preprocessing (Encoding, Imputation, Conversion)
# =============================================================================

def create_initial_pipeline(X, one_hot_features, ordinal_features, scaled_features):
    # Compute other numerical features from X
    all_numeric = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    other_num_features = [col for col in all_numeric if col not in scaled_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary'), one_hot_features),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_features),
            ('numerical', KNNImputer(n_neighbors=5), scaled_features),
            ('others', 'passthrough', other_num_features)
        ],
        remainder='drop'
    )
    
    # Fit preprocessor to obtain one-hot output names
    preprocessor.fit(X)
    preprocessed_feature_names = []
    if one_hot_features:
        onehot_names = preprocessor.named_transformers_['onehot'].get_feature_names_out(one_hot_features)
        preprocessed_feature_names.extend(onehot_names)
    preprocessed_feature_names.extend(ordinal_features)
    preprocessed_feature_names.extend(scaled_features)
    preprocessed_feature_names.extend(other_num_features)
    
    initial_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('to_dataframe', DataFrameConverter(columns=preprocessed_feature_names))
    ])
    
    # Fit the initial pipeline
    initial_pipeline.fit(X)
    return initial_pipeline

# =============================================================================
# Second Pipeline: Post-Processing (Dropping, Interactions, VIF)
# =============================================================================

def create_post_pipeline():
    post_pipeline = Pipeline(steps=[
        ('drop_turbo', FunctionTransformer(drop_turbo_column, validate=False)),
        ('interactions', InteractionCreator()),
        ('vif_reduction', VIFReducer(thresh=10.0))
    ])
    return post_pipeline

# =============================================================================
# Combined Pipeline Function
# =============================================================================

def create_full_pipeline(X, one_hot_features, ordinal_features, scaled_features):
    # Create and fit the initial pipeline
    initial_pipeline = create_initial_pipeline(X, one_hot_features, ordinal_features, scaled_features)
    # Transform the data to get a DataFrame with proper column names
    X_initial = initial_pipeline.transform(X)
    # Create and fit the post pipeline on the already-transformed data
    post_pipeline = create_post_pipeline()
    post_pipeline.fit(X_initial)
    # Combine both pipelines into one final pipeline using Pipeline chaining
    full_pipeline = Pipeline(steps=[
        ('initial', initial_pipeline),
        ('post', post_pipeline)
    ])
    return full_pipeline

# =============================================================================
# Example Usage
# =============================================================================

if __name__ == '__main__':
    # Load your training data
    X_train = pd.read_csv("path_to_training_data.csv")
    
    # Define your feature lists:
    one_hot_features = [...]   # e.g., list of categorical features for one-hot encoding
    ordinal_features = [...]     # e.g., list of features for ordinal encoding
    scaled_features = [...]      # e.g., list of numerical features to impute/scale
    
    # Create the full pipeline
    full_pipeline = create_full_pipeline(X_train, one_hot_features, ordinal_features, scaled_features)
    
    # Now you can transform your training (or any new) data in one call:
    X_train_transformed = full_pipeline.fittransform(X_train)
    print("Transformed training data shape:", X_train_transformed.shape)
