from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import time

# Start Spark session
spark = SparkSession.builder.appName("web-page-phishing").getOrCreate()

# Load the dataset
file_path = "/opt/spark/web-page-phishing.csv"
data = spark.read.csv(file_path, header=True, inferSchema=True)

# Select relevant features and the target variable
features = [col for col in data.columns if col != 'phishing']
target = 'phishing'

# Index the target variable
indexer = StringIndexer(inputCol=target, outputCol="label")
data = indexer.fit(data).transform(data)

# Assemble features into a single vector column
assembler = VectorAssembler(inputCols=features, outputCol="features")
data = assembler.transform(data)

# Scale features for SVM
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
data = scaler.fit(data).transform(data)

# Split dataset into training and testing sets
train, test = data.randomSplit([0.8, 0.2], seed=42)

# Function to evaluate model performance
def evaluate_model(predictions, model_name):
    metrics = {}
    
    # Multiclass classification metrics
    accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
    precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    
    # Binary classification metrics
    auc_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

    # Store evaluation metrics
    metrics['Accuracy'] = accuracy_evaluator.evaluate(predictions)
    metrics['Recall'] = recall_evaluator.evaluate(predictions)
    metrics['Precision'] = precision_evaluator.evaluate(predictions)
    metrics['AUC'] = auc_evaluator.evaluate(predictions)
    
    # Print evaluation results
    print(f"Performance of {model_name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    print("\n")
    
    return metrics

# Random Forest model
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100)
rf_model = rf.fit(train)
rf_predictions = rf_model.transform(test)
rf_metrics = evaluate_model(rf_predictions, "Random Forest")

# Support Vector Machine (SVM) model
svm = LinearSVC(featuresCol="scaledFeatures", labelCol="label", maxIter=10, regParam=0.1)
svm_model = svm.fit(train)
svm_predictions = svm_model.transform(test)
svm_metrics = evaluate_model(svm_predictions, "SVM")

# Compare model performances
print("Model Comparison:")
print(f"{'Metric':<10}{'Random Forest':<15}{'SVM':<15}")
for metric in rf_metrics:
    print(f"{metric:<10}{rf_metrics[metric]:<15.4f}{svm_metrics[metric]:<15.4f}")

# Stop Spark session
spark.stop()
