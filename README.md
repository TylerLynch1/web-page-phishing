# web-page-phishing

This project addresses phishing detection by training machine learning models to classify websites as phishing or legitimate. Using a dataset of website characteristics, the script preprocessed data by indexing the target variable, assembling features into a vector, and scaling features for SVM models.
The code employed PySpark libraries, specifically SparkSession, StringIndexer, VectorAssembler, StandardScaler, and Spark's MLlib for model training and evaluation. Two models were trained: a Random Forest Classifier and an SVM, chosen for their effectiveness in binary classification. The dataset was split into training and testing sets, with each model evaluated for accuracy, recall, precision, and AUC.
