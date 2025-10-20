import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import save_numpy_array_data, save_object, read_csv_file, read_yaml_file

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "data_transformation", "preprocessor.pkl")
    transformed_train_file_path: str = os.path.join("artifacts", "data_transformation", "train.npy")
    transformed_test_file_path: str = os.path.join("artifacts", "data_transformation", "test.npy")

class DataTransformation:
    SCHEMA_PATH = os.path.join("config", "schema.yaml")

    def __init__(self):
        try:
            self.config = DataTransformationConfig()
            os.makedirs(os.path.dirname(self.config.preprocessor_obj_file_path), exist_ok=True)

            # Read schema
            self.schema = read_yaml_file(self.SCHEMA_PATH)
            self.numerical_columns = self.schema.get("numerical_columns", [])
            self.ohe_columns = self.schema.get("ohe_columns", [])
            self.target_column = self.schema.get("target_column")
            
            logging.info(f"Schema loaded successfully. Numerical: {self.numerical_columns}, "
                         f"Target: {self.target_column}")
        except Exception as e:
            logging.error("Error initializing DataTransformation with schema")
            raise CustomException(e, sys)

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Type conversion
            df[["Fertilizer_Used", "Irrigation_Used"]] = df[["Fertilizer_Used", "Irrigation_Used"]].astype(int)
            df[self.target_column] = df[self.target_column].clip(lower=0)
            return df
        except Exception as e:
            logging.error("Error in feature engineering")
            raise CustomException(e, sys)

    def get_preprocessor_pipeline(self):
        try:
            # ColumnTransformer
            preprocessor = ColumnTransformer(transformers=[
                ('num', StandardScaler(), self.numerical_columns),
                ('ohe', OneHotEncoder(drop='first', sparse_output=False), self.ohe_columns)
            ], remainder="passthrough")

            return preprocessor
        except Exception as e:
            logging.error("Error creating preprocessor pipeline")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            logging.info("Reading train and test data")
            train_df = read_csv_file(train_path)
            test_df = read_csv_file(test_path)

            logging.info("Applying feature engineering on train and test data")
            train_df = self.feature_engineering(train_df)
            test_df = self.feature_engineering(test_df)

            X_train = train_df.drop(columns=[self.target_column], axis=1)
            y_train = train_df[self.target_column]
            X_test = test_df.drop(columns=[self.target_column], axis=1)
            y_test = test_df[self.target_column]

            logging.info("Creating preprocessing pipeline")
            preprocessor = self.get_preprocessor_pipeline()

            logging.info("Fitting preprocessor on training data")
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            logging.info("Saving transformed data and preprocessor")
            save_numpy_array_data(self.config.transformed_train_file_path, 
                                  np.c_[X_train_transformed, y_train.values.reshape(-1, 1)])
            save_numpy_array_data(self.config.transformed_test_file_path, 
                                  np.c_[X_test_transformed, y_test.values.reshape(-1, 1)])
            save_object(self.config.preprocessor_obj_file_path, preprocessor)
            
            logging.info(f"Data transformation completed and saved successfully at {self.config.preprocessor_obj_file_path}")
            return X_train_transformed, X_test_transformed, y_train, y_test
        except Exception as e:
            logging.error("Error in data transformation")
            raise CustomException(e, sys)


# ---- Testing ----
if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion, DataIngestionConfig

    # Paths to train and test data
    ingest_config = DataIngestionConfig()
    data_ingestion = DataIngestion(config=ingest_config)
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    # Initialize the transformer
    transformer = DataTransformation()

    # Run the data transformation
    X_train_transformed, X_test_transformed, y_train, y_test = transformer.initiate_data_transformation(
        train_path=train_data_path,
        test_path=test_data_path
    )

    # Quick checks
    print("Transformed X_train shape:", X_train_transformed.shape)
    print("Transformed X_test shape:", X_test_transformed.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)