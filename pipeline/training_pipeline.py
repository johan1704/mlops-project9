from src.data_processing import DataProcessing
from src.model_training import ModelTraining
if __name__ == "__main__":
    processor = DataProcessing(
        input_path="artifacts/raw/data.csv",
        output_path="artifacts/processed"
    )
    processor.run()

    trainer = ModelTraining(
        input_path="artifacts/processed",
        output_path="artifacts/models"
    )
    trainer.run()