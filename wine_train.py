from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col

def create_spark_session():
    """Create Spark session for distributed training"""
    return SparkSession.builder \
        .appName("WineQualityTraining") \
        .master("spark://MASTER-IP:7077") \
        .config("spark.executor.memory", "2g") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

def prepare_data(spark, file_path):
    """Load and prepare data for training"""
    # Read CSV file
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    
    # Select feature columns (all columns except 'quality')
    feature_cols = [col for col in df.columns if col != 'quality']
    
    # Create feature vector
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    data = assembler.transform(df).select("features", col("quality").cast("double").alias("label"))
    
    return data

def train_model():
    """Train the wine quality prediction model"""
    # Initialize Spark
    spark = create_spark_session()
    
    # Define file paths
    training_path = "/home/ec2-user/job/TrainingDataset.csv"
    validation_path = "/home/ec2-user/job/ValidationDataset.csv"
    model_save_path = "/home/ec2-user/job/wine_model"
    
    # Load and prepare training data
    print("Loading training data...")
    training_data = prepare_data(spark, training_path)
    validation_data = prepare_data(spark, validation_path)
    
    # Initialize and train logistic regression model
    print("Training model...")
    lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    model = lr.fit(training_data)
    
    # Make predictions on training dataset
    print("Evaluating model on training data...")
    train_predictions = model.transform(training_data)
    
    # Evaluate the model
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", 
        predictionCol="prediction", 
        metricName="f1"
    )
    f1_score = evaluator.evaluate(train_predictions)

    # Calculate accuracy
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", 
        predictionCol="prediction", 
        metricName="accuracy"
    )
    accuracy = evaluator.evaluate(train_predictions)
    
    # Output the results
    print("[Train] F1 score =", f1_score)
    print("[Train] Accuracy =", accuracy)
    
    # Save the model
    print(f"Saving model to {model_save_path}")
    model.save(model_save_path)
    
    spark.stop()

if __name__ == "__main__":
    train_model()
