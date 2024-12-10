import logging
from pathlib import Path
from typing import List, Tuple

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WineQualityPredictor:
    def __init__(self, model_dir: str = "/home/ec2-user/models"):
        """Initialize the Wine Quality Predictor.

        Args:
            model_dir: Directory for saving model and results
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / "wine_model"
        
    def create_spark_session(self) -> SparkSession:
        """Create and configure Spark session."""
        return SparkSession.builder \
            .appName("WineQualityPredictor") \
            .master("local[*]") \
            .getOrCreate()

    def load_data(self, spark: SparkSession, test_path: str) -> DataFrame:
        """Load and prepare test dataset.

        Args:
            spark: Active SparkSession
            test_path: Path to test dataset

        Returns:
            Test DataFrame
        """
        logger.info("Loading test dataset...")
        data_test = spark.read.option("delimiter", ";") \
            .csv(test_path, header=True, inferSchema=True)
        return self._clean_data(data_test)

    def _clean_data(self, df: DataFrame) -> DataFrame:
        """Clean column names and filter unwanted values.

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        for old_name in df.schema.names:
            clean_name = old_name.replace('"', '')
            if clean_name != old_name:
                df = df.withColumnRenamed(old_name, clean_name)
        return df.filter(df["quality"] != 3)

    def evaluate_model(self, predictions: DataFrame) -> Tuple[float, float]:
        """Calculate F1 score and accuracy for predictions.

        Args:
            predictions: DataFrame with predictions

        Returns:
            Tuple of (f1_score, accuracy)
        """
        evaluator = MulticlassClassificationEvaluator(
            labelCol="quality", 
            predictionCol="prediction"
        )
        f1_score = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
        accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
        return f1_score, accuracy

    def test_model(self, test_path: str):
        """Main method to test the model on new data.

        Args:
            test_path: Path to test dataset
        """
        try:
            spark = self.create_spark_session()

            # Load test data
            data_test = self.load_data(spark, test_path)
            
            # Load saved model
            logger.info(f"Loading model from {self.model_path}")
            model = PipelineModel.load(str(self.model_path))
            
            # Generate predictions
            logger.info("Generating predictions...")
            test_predictions = model.transform(data_test)
            
            # Evaluate model
            test_f1_score, test_accuracy = self.evaluate_model(test_predictions)
            
            # Output results
            print("[Test] F1 score =", test_f1_score)
            print("[Test] Accuracy =", test_accuracy)

        except Exception as e:
            logger.error(f"Error occurred: {str(e)}", exc_info=True)
            raise

        finally:
            spark.stop()

def main():
    """Main entry point of the application."""
    predictor = WineQualityPredictor()
    predictor.test_model(test_path="ValidationDataset.csv")

if __name__ == "__main__":
    main()
