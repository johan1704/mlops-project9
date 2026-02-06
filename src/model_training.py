import os
import joblib
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
from sklearn.model_selection import cross_val_score
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, input_path, output_path, model_params=None):
        """
        Initialize model training
        
        Args:
            input_path: Path to processed data
            output_path: Path to save model and artifacts
            model_params: Dict of XGBoost parameters (optional)
        """
        self.input_path = input_path
        self.output_path = output_path
        
        # Default XGBoost parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        
        # Use custom params or defaults
        self.model_params = model_params if model_params else default_params
        self.model = xgb.XGBClassifier(**self.model_params)
        
        # Data placeholders
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Metrics placeholder
        self.metrics = {}
        
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path, 'plots'), exist_ok=True)
        
        logger.info("ModelTraining initialized...")
        logger.info(f"Model parameters: {self.model_params}")
    
    def load_data(self):
        """Load preprocessed train/test data"""
        try:
            self.X_train = joblib.load(os.path.join(self.input_path, "X_train.pkl"))
            self.X_test = joblib.load(os.path.join(self.input_path, "X_test.pkl"))
            self.y_train = joblib.load(os.path.join(self.input_path, "y_train.pkl"))
            self.y_test = joblib.load(os.path.join(self.input_path, "y_test.pkl"))
            
            logger.info(f"Data loaded successfully...")
            logger.info(f"Train set: {self.X_train.shape}")
            logger.info(f"Test set: {self.X_test.shape}")
            logger.info(f"Features: {self.X_train.columns.tolist()}")
            
            return self
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise CustomException("Failed to load data", e)
    
    def train_model(self):
        """Train XGBoost model"""
        try:
            logger.info("Starting model training...")
            
            # Train model
            self.model.fit(
                self.X_train,
                self.y_train,
                eval_set=[(self.X_test, self.y_test)],
                verbose=False
            )
            
            # Save model
            model_path = os.path.join(self.output_path, "model.pkl")
            joblib.dump(self.model, model_path)
            
            logger.info(f"Model trained and saved to {model_path}")
            return self
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise CustomException("Failed to train model", e)
    
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        try:
            logger.info("Evaluating model...")
            
            # Predictions
            y_pred = self.model.predict(self.X_test)
            y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
            
            # Training score
            train_score = self.model.score(self.X_train, self.y_train)
            
            # Test metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average="weighted")
            recall = recall_score(self.y_test, y_pred, average="weighted")
            f1 = f1_score(self.y_test, y_pred, average="weighted")
            
            # ROC-AUC (if binary classification)
            if len(np.unique(self.y_train)) == 2:
                roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            else:
                roc_auc = None
            
            # Cross-validation score
            cv_scores = cross_val_score(
                self.model, self.X_train, self.y_train,
                cv=5, scoring='accuracy'
            )
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Store metrics
            self.metrics = {
                'train_score': float(train_score),
                'test_accuracy': float(accuracy),
                'test_precision': float(precision),
                'test_recall': float(recall),
                'test_f1_score': float(f1),
                'test_roc_auc': float(roc_auc) if roc_auc else None,
                'cv_mean_accuracy': float(cv_mean),
                'cv_std_accuracy': float(cv_std),
                'model_params': self.model_params,
                'training_date': datetime.now().isoformat(),
                'n_features': self.X_train.shape[1],
                'n_train_samples': self.X_train.shape[0],
                'n_test_samples': self.X_test.shape[0]
            }
            
            # Log metrics
            logger.info("=" * 60)
            logger.info("MODEL PERFORMANCE METRICS")
            logger.info("=" * 60)
            logger.info(f"Training Score:        {train_score:.4f}")
            logger.info(f"Test Accuracy:         {accuracy:.4f}")
            logger.info(f"Test Precision:        {precision:.4f}")
            logger.info(f"Test Recall:           {recall:.4f}")
            logger.info(f"Test F1-Score:         {f1:.4f}")
            if roc_auc:
                logger.info(f"Test ROC-AUC:          {roc_auc:.4f}")
            logger.info(f"CV Accuracy (mean±std): {cv_mean:.4f} ± {cv_std:.4f}")
            logger.info("=" * 60)
            
            # Classification report
            logger.info("\nDetailed Classification Report:")
            logger.info("\n" + classification_report(self.y_test, y_pred))
            
            # Save metrics to JSON
            metrics_path = os.path.join(self.output_path, "metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=4)
            logger.info(f"Metrics saved to {metrics_path}")
            
            return self
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise CustomException("Failed to evaluate model", e)
    
    def plot_confusion_matrix(self):
        """Plot and save confusion matrix"""
        try:
            y_pred = self.model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
            plt.title('Confusion Matrix')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            
            cm_path = os.path.join(self.output_path, 'plots', 'confusion_matrix.png')
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Confusion matrix saved to {cm_path}")
            return self
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")
            raise CustomException("Failed to plot confusion matrix", e)
    
    def plot_roc_curve(self):
        """Plot and save ROC curve"""
        try:
            # Only for binary classification
            if len(np.unique(self.y_train)) != 2:
                logger.info("ROC curve only for binary classification, skipping...")
                return self
            
            y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                    label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            
            roc_path = os.path.join(self.output_path, 'plots', 'roc_curve.png')
            plt.savefig(roc_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"ROC curve saved to {roc_path}")
            return self
            
        except Exception as e:
            logger.error(f"Error plotting ROC curve: {e}")
            raise CustomException("Failed to plot ROC curve", e)
    
    def plot_feature_importance(self):
        """Plot and save feature importance"""
        try:
            # Get feature importance
            importance = self.model.feature_importances_
            features = self.X_train.columns
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': features,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # Save to CSV
            importance_path = os.path.join(self.output_path, 'feature_importance.csv')
            importance_df.to_csv(importance_path, index=False)
            logger.info(f"Feature importance saved to {importance_path}")
            
            # Plot top 20 features
            plt.figure(figsize=(10, 8))
            top_n = min(20, len(importance_df))
            top_features = importance_df.head(top_n)
            
            plt.barh(range(top_n), top_features['importance'])
            plt.yticks(range(top_n), top_features['feature'])
            plt.xlabel('Importance')
            plt.title(f'Top {top_n} Feature Importances')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            plot_path = os.path.join(self.output_path, 'plots', 'feature_importance.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Feature importance plot saved to {plot_path}")
            
            # Log top 10
            logger.info("\nTop 10 Most Important Features:")
            for idx, row in importance_df.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            return self
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")
            raise CustomException("Failed to plot feature importance", e)
    
    def save_model_metadata(self):
        """Save model metadata for production use"""
        try:
            metadata = {
                'model_type': 'XGBClassifier',
                'model_version': '1.0.0',
                'training_date': datetime.now().isoformat(),
                'features': self.X_train.columns.tolist(),
                'n_features': self.X_train.shape[1],
                'hyperparameters': self.model_params,
                'metrics': self.metrics,
                'input_data_path': self.input_path,
                'model_path': os.path.join(self.output_path, 'model.pkl')
            }
            
            metadata_path = os.path.join(self.output_path, 'model_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Model metadata saved to {metadata_path}")
            return self
            
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            raise CustomException("Failed to save metadata", e)
    
    def run(self):
        """Run complete training pipeline"""
        try:
            logger.info("=" * 60)
            logger.info("STARTING MODEL TRAINING PIPELINE")
            logger.info("=" * 60)
            
            self.load_data()
            self.train_model()
            self.evaluate_model()
            self.plot_confusion_matrix()
            self.plot_roc_curve()
            self.plot_feature_importance()
            self.save_model_metadata()
            
            logger.info("=" * 60)
            logger.info("MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise

if __name__ == "__main__":
    trainer = ModelTraining(
        input_path="artifacts/processed",
        output_path="artifacts/models"
    )
    trainer.run()