import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class DataProcessing:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.df = None
        self.encoders = {}
        self.target_encoder = None
        self.fill_values = {}
        self.numerical_cols = []
        self.categorical_cols = []
        
        os.makedirs(self.output_path, exist_ok=True)
        logger.info("DataProcessing initialized...")
    
    def load_data(self):
        """Load raw data from CSV"""
        try:
            self.df = pd.read_csv(self.input_path)
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise CustomException("Failed to load data", e)
    
    def validate_data(self):
        """Validate data has expected columns"""
        try:
            expected_columns = [
                'Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall',
                'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed',
                'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm',
                'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
                'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm',
                'RainToday', 'RainTomorrow'
            ]
            
            missing_cols = set(expected_columns) - set(self.df.columns)
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")
            
            logger.info("Data validation passed...")
            return self
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            raise CustomException("Data validation failed", e)
    
    def preprocess(self):
        """Preprocess data: handle dates, missing values"""
        try:
            # Define column types
            self.numerical_cols = [
                'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
                'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
                'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
                'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm'
            ]
            
            self.categorical_cols = [
                'Location', 'WindGustDir', 'WindDir9am',
                'WindDir3pm', 'RainToday'
            ]
            
            # Process date
            self.df["Date"] = pd.to_datetime(self.df["Date"])
            self.df["Year"] = self.df["Date"].dt.year
            self.df["Month"] = self.df["Date"].dt.month
            self.df["Day"] = self.df["Date"].dt.day
            self.df.drop("Date", axis=1, inplace=True)
            
            # Handle missing values and save statistics
            for col in self.numerical_cols:
                if col in self.df.columns:
                    mean_value = self.df[col].mean()
                    self.df[col].fillna(mean_value, inplace=True)
                    self.fill_values[col] = mean_value
            
            # Save fill values
            joblib.dump(
                self.fill_values,
                os.path.join(self.output_path, "fill_values.pkl")
            )
            
            # Drop remaining NaN
            initial_rows = len(self.df)
            self.df.dropna(inplace=True)
            dropped_rows = initial_rows - len(self.df)
            
            logger.info(f"Preprocessing done. Dropped {dropped_rows} rows with NaN...")
            return self
            
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            raise CustomException("Failed to preprocess data", e)
    
    def label_encode(self):
        """Label encode categorical features and target"""
        try:
            # Encode categorical features
            for col in self.categorical_cols:
                if col in self.df.columns:
                    encoder = LabelEncoder()
                    self.df[col] = encoder.fit_transform(self.df[col])
                    self.encoders[col] = encoder
                    
                    label_mapping = dict(zip(
                        encoder.classes_,
                        range(len(encoder.classes_))
                    ))
                    logger.info(f"Label mapping for {col}: {label_mapping}")
            
            # Encode target separately
            self.target_encoder = LabelEncoder()
            self.df['RainTomorrow'] = self.target_encoder.fit_transform(
                self.df['RainTomorrow']
            )
            
            target_mapping = dict(zip(
                self.target_encoder.classes_,
                range(len(self.target_encoder.classes_))
            ))
            logger.info(f"Target mapping: {target_mapping}")
            
            # Save encoders
            joblib.dump(
                self.encoders,
                os.path.join(self.output_path, "feature_encoders.pkl")
            )
            joblib.dump(
                self.target_encoder,
                os.path.join(self.output_path, "target_encoder.pkl")
            )
            
            logger.info("Label encoding completed and saved...")
            return self
            
        except Exception as e:
            logger.error(f"Error during label encoding: {e}")
            raise CustomException("Failed to label encode", e)
    
    def split_data(self):
        """Split data into train/test sets"""
        try:
            # Save processed data
            self.df.to_csv(
                os.path.join(self.output_path, "processed_data.csv"),
                index=False
            )
            
            # Split features and target
            X = self.df.drop('RainTomorrow', axis=1)
            y = self.df["RainTomorrow"]
            
            logger.info(f"Features ({len(X.columns)}): {X.columns.tolist()}")
            logger.info(f"Target distribution:\n{y.value_counts()}")
            
            # Train/test split with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )
            
            # Save splits
            joblib.dump(X_train, os.path.join(self.output_path, "X_train.pkl"))
            joblib.dump(X_test, os.path.join(self.output_path, "X_test.pkl"))
            joblib.dump(y_train, os.path.join(self.output_path, "y_train.pkl"))
            joblib.dump(y_test, os.path.join(self.output_path, "y_test.pkl"))
            
            logger.info(f"Train set: {len(X_train)} samples")
            logger.info(f"Test set: {len(X_test)} samples")
            logger.info("Data split and saved successfully...")
            return self
            
        except Exception as e:
            logger.error(f"Error during data splitting: {e}")
            raise CustomException("Failed to split data", e)
    
    def run(self):
        """Run complete data processing pipeline"""
        try:
            self.load_data()
            self.validate_data()
            self.preprocess()
            self.label_encode()
            self.split_data()
            logger.info("=" * 50)
            logger.info("DATA PROCESSING COMPLETED SUCCESSFULLY")
            logger.info("=" * 50)
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

if __name__ == "__main__":
    processor = DataProcessing(
        input_path="artifacts/raw/data.csv",
        output_path="artifacts/processed"
    )
    processor.run()