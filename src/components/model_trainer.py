# src/components/model_trainer.py
import os, sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import save_object

@dataclass
class ModelTrainerConfig:
    model_file_path: str = os.path.join("artifacts", "model_trainer", "reg.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        os.makedirs(os.path.dirname(self.config.model_file_path), exist_ok=True)
        logging.info(f"ModelTrainer initialized. Model will be saved at {self.config.model_file_path}")

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        """
        Trains a Linear regression model and evaluates it.
        Saves the trained model as a pickle file.
        """
        try:
            logging.info("Model training started")
            # Initialize model
            reg = LinearRegression()
            logging.info("LinearRegression model instance created")

            # Train model
            reg.fit(X_train, y_train)
            logging.info("Model training completed")

            # Predict
            logging.info("Predicting on training and test data")
            y_train_pred = reg.predict(X_train)
            y_test_pred = reg.predict(X_test)

            # Evaluate
            logging.info("Evaluating model performance")
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = root_mean_squared_error(y_train, y_train_pred)
            test_rmse = root_mean_squared_error(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)

            logging.info(f"Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}")
            logging.info(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
            logging.info(f"Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")

            # Save model
            save_object(self.config.model_file_path, reg)
            logging.info(f"Trained model saved at: {self.config.model_file_path}")
            return reg

        except Exception as e:
            logging.error("Error in model training")
            raise CustomException(e, sys)
        


# ---- Testing ----
if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion, DataIngestionConfig
    from src.components.data_transformation import DataTransformation

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

    model = ModelTrainer()
    model.initiate_model_trainer(X_train_transformed, X_test_transformed, y_train, y_test)