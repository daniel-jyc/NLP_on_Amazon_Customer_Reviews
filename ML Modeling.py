#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sparknlp
import logging
sparknlp.start()
import numpy as np

from sparknlp import *
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.pretrained import PretrainedPipeline

from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, udf
from pyspark.sql.types import *

from pyspark.ml import Pipeline
from pyspark.ml.feature import *
from pyspark.ml.regression import *
from pyspark.ml.classification import *
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# ### Create Spark Session

# In[2]:


spark = SparkSession.builder.appName('Spark-Sentiment').config("spark.sql.autoBroadcastJoinThreshold", -1).getOrCreate()
logger = spark.sparkContext._jvm.org.apache.log4j
logger.LogManager.getLogger("org.apache.spark.scheduler").setLevel(logger.Level.ERROR)
logging.getLogger("py4j").setLevel(logging.ERROR)
spark.sparkContext.setLogLevel("ERROR")


# In[3]:


spark.conf.set("spark.hadoop.google.cloud.auth.service.account.enable", "true")
spark.conf.set("spark.hadoop.google.cloud.auth.service.account.json.keyfile", "path/to/your/credentials.json")


# ### Pipeline Construction

# In[4]:


tokenizer = Tokenizer(inputCol="review_body", outputCol="review_body_words")
remover = StopWordsRemover(inputCol="review_body_words", outputCol="review_body_words_filtered")
hashingTF = HashingTF(inputCol="review_body_words_filtered", outputCol="hashingTF_features")
idf = IDF(inputCol="hashingTF_features", outputCol="idf_features")
labelIndexer = StringIndexer(inputCol="sentiment", outputCol="sentiment_label")

pipeline = Pipeline(stages=[tokenizer,remover,hashingTF,idf,labelIndexer])


# ### Dataset cleaning and spliting

# In[6]:


# ## Apparel
df_apparel = spark.read.format("csv").option("header", "true").                option("delimiter", "\t").load("gs://msca-bdp-student-gcs/Group4_Project_Data/amazon_reviews_us_Apparel_v1_00.tsv")
df_apparel = df_apparel.dropna().withColumn("star_rating",df_apparel.star_rating.cast('int')).withColumn('sentiment', when(col('star_rating') <= 4, 'negative').otherwise('positive'))

result_apparel = pipeline.fit(df_apparel).transform(df_apparel).select('review_id','product_id','product_title','sentiment_label','idf_features')
trainData_apparel, testData_apparel = result_apparel.randomSplit([0.7,0.3])

nb_apparel = NaiveBayes(smoothing=1.0, modelType="multinomial", featuresCol = 'idf_features', labelCol='sentiment_label').fit(trainData_apparel)
nb_predictions_apparel = nb_apparel.transform(testData_apparel)
evaluator = MulticlassClassificationEvaluator(labelCol='sentiment_label', predictionCol="prediction", metricName="accuracy")
nb_accuracy_apparel = evaluator.evaluate(nb_predictions_apparel)
print("Accuracy = %g" % (nb_accuracy_apparel))


# In[7]:


# ## Automotive
df_automotive = spark.read.format("csv").option("header", "true").                option("delimiter", "\t").load("gs://msca-bdp-student-gcs/Group4_Project_Data/amazon_reviews_us_Automotive_v1_00.tsv")
df_automotive = df_automotive.dropna().withColumn("star_rating",df_automotive.star_rating.cast('int')).withColumn('sentiment', when(col('star_rating') <= 4, 'negative').otherwise('positive'))

result_automotive = pipeline.fit(df_automotive).transform(df_automotive).select('review_id','product_id','product_title','sentiment_label','idf_features')
trainData_automotive, testData_automotive = result_automotive.randomSplit([0.7,0.3])

nb_automotive = NaiveBayes(smoothing=1.0, modelType="multinomial", featuresCol = 'idf_features', labelCol='sentiment_label').fit(trainData_automotive)
nb_predictions_automotive = nb_automotive.transform(testData_automotive)
evaluator = MulticlassClassificationEvaluator(labelCol='sentiment_label', predictionCol="prediction", metricName="accuracy")
nb_accuracy_automotive = evaluator.evaluate(nb_predictions_automotive)
print("Accuracy = %g" % (nb_accuracy_automotive))


# In[7]:


## Beauty
df_beauty = spark.read.format("csv").option("header", "true").                option("delimiter", "\t").load("gs://msca-bdp-student-gcs/Group4_Project_Data/amazon_reviews_us_Beauty_v1_00.tsv")
df_beauty = df_beauty.dropna().withColumn("star_rating",df_beauty.star_rating.cast('int')).withColumn('sentiment', when(col('star_rating') <= 3, 'negative').otherwise('positive'))

result_beauty = pipeline.fit(df_beauty).transform(df_beauty).select('review_id','product_id','product_title','sentiment_label','idf_features')
trainData_beauty, testData_beauty = result_beauty.randomSplit([0.7,0.3])

nb_beauty = NaiveBayes(smoothing=1.0, modelType="multinomial", featuresCol = 'idf_features', labelCol='sentiment_label').fit(trainData_beauty)
nb_predictions_beauty = nb_beauty.transform(testData_beauty)
evaluator = MulticlassClassificationEvaluator(labelCol='sentiment_label', predictionCol="prediction", metricName="accuracy")
nb_accuracy_beauty = evaluator.evaluate(nb_predictions_beauty)
print("Accuracy = %g" % (nb_accuracy_beauty))


# In[8]:


## Electronics
df_electronics = spark.read.format("csv").option("header", "true").                option("delimiter", "\t").load("gs://msca-bdp-student-gcs/Group4_Project_Data/amazon_reviews_us_Electronics_v1_00.tsv")
df_electronics = df_electronics.dropna().withColumn("star_rating",df_electronics.star_rating.cast('int')).withColumn('sentiment', when(col('star_rating') <= 3, 'negative').otherwise('positive'))

result_electronics = pipeline.fit(df_electronics).transform(df_electronics).select('review_id','product_id','product_title','sentiment_label','idf_features')
trainData_electronics, testData_electronics = result_electronics.randomSplit([0.7,0.3])

nb_electronics = NaiveBayes(smoothing=1.0, modelType="multinomial", featuresCol = 'idf_features', labelCol='sentiment_label').fit(trainData_electronics)
nb_predictions_electronics = nb_electronics.transform(testData_electronics)
evaluator = MulticlassClassificationEvaluator(labelCol='sentiment_label', predictionCol="prediction", metricName="accuracy")
nb_accuracy_electronics = evaluator.evaluate(nb_predictions_electronics)
print("Accuracy = %g" % (nb_accuracy_electronics))


# In[9]:


## Shoes
df_shoes = spark.read.format("csv").option("header", "true").                option("delimiter", "\t").load("gs://msca-bdp-student-gcs/Group4_Project_Data/amazon_reviews_us_Shoes_v1_00.tsv")
df_shoes = df_shoes.dropna().withColumn("star_rating",df_shoes.star_rating.cast('int')).withColumn('sentiment', when(col('star_rating') <= 3, 'negative').otherwise('positive'))

result_shoes = pipeline.fit(df_shoes).transform(df_shoes).select('review_id','product_id','product_title','sentiment_label','idf_features')
trainData_shoes, testData_shoes = result_shoes.randomSplit([0.7,0.3])
nb_shoes = NaiveBayes(smoothing=1.0, modelType="multinomial", featuresCol = 'idf_features', labelCol='sentiment_label').fit(trainData_shoes)
nb_predictions_shoes = nb_shoes.transform(testData_shoes)
evaluator = MulticlassClassificationEvaluator(labelCol='sentiment_label', predictionCol="prediction", metricName="accuracy")
nb_accuracy_shoes = evaluator.evaluate(nb_predictions_shoes)
print("Accuracy = %g" % (nb_accuracy_shoes))


# In[10]:


## Pet Products
df_pet_products = spark.read.format("csv").option("header", "true").                option("delimiter", "\t").load("gs://msca-bdp-student-gcs/Group4_Project_Data/amazon_reviews_us_Pet_Products_v1_00.tsv")
df_pet_products = df_pet_products.dropna().withColumn("star_rating",df_pet_products.star_rating.cast('int')).withColumn('sentiment', when(col('star_rating') <= 3, 'negative').otherwise('positive'))

result_pet_products = pipeline.fit(df_pet_products).transform(df_pet_products).select('review_id','product_id','product_title','sentiment_label','idf_features')
trainData_pet_products, testData_pet_products = result_pet_products.randomSplit([0.7,0.3])

nb_pet_products = NaiveBayes(smoothing=1.0, modelType="multinomial", featuresCol = 'idf_features', labelCol='sentiment_label').fit(trainData_pet_products)
nb_predictions_pet_products = nb_pet_products.transform(testData_pet_products)
evaluator = MulticlassClassificationEvaluator(labelCol='sentiment_label', predictionCol="prediction", metricName="accuracy")
nb_accuracy_pet_products = evaluator.evaluate(nb_predictions_pet_products)
print("Accuracy = %g" % (nb_accuracy_pet_products))


# In[11]:


## Sports
df_sports = spark.read.format("csv").option("header", "true").                option("delimiter", "\t").load("gs://msca-bdp-student-gcs/Group4_Project_Data/amazon_reviews_us_Sports_v1_00.tsv")
df_sports = df_sports.dropna().withColumn("star_rating",df_sports.star_rating.cast('int')).withColumn('sentiment', when(col('star_rating') <= 3, 'negative').otherwise('positive'))

result_sports = pipeline.fit(df_sports).transform(df_sports).select('review_id','product_id','product_title','sentiment_label','idf_features')
trainData_sports, testData_sports = result_sports.randomSplit([0.7,0.3])

nb_sports = NaiveBayes(smoothing=1.0, modelType="multinomial", featuresCol = 'idf_features', labelCol='sentiment_label').fit(trainData_sports)
nb_predictions_sports = nb_sports.transform(testData_sports)
evaluator = MulticlassClassificationEvaluator(labelCol='sentiment_label', predictionCol="prediction", metricName="accuracy")
nb_accuracy_sports = evaluator.evaluate(nb_predictions_sports)
print("Accuracy = %g" % (nb_accuracy_sports))


# In[12]:


## Toys
df_toys = spark.read.format("csv").option("header", "true").                option("delimiter", "\t").load("gs://msca-bdp-student-gcs/Group4_Project_Data/amazon_reviews_us_Toys_v1_00.tsv")
df_toys = df_toys.dropna().withColumn("star_rating",df_toys.star_rating.cast('int')).withColumn('sentiment', when(col('star_rating') <= 3, 'negative').otherwise('positive'))

result_toys = pipeline.fit(df_toys).transform(df_toys).select('review_id','product_id','product_title','sentiment_label','idf_features')
trainData_toys, testData_toys = result_toys.randomSplit([0.7,0.3])

nb_toys = NaiveBayes(smoothing=1.0, modelType="multinomial", featuresCol = 'idf_features', labelCol='sentiment_label').fit(trainData_toys)
nb_predictions_toys = nb_toys.transform(testData_toys)
evaluator = MulticlassClassificationEvaluator(labelCol='sentiment_label', predictionCol="prediction", metricName="accuracy")
nb_accuracy_toys = evaluator.evaluate(nb_predictions_toys)
print("Accuracy = %g" % (nb_accuracy_toys))


# ### Save the models to GCP Storage

# In[13]:


from pyspark.ml.util import MLWritable

bucket_name = "msca-bdp-student-gcs"

model_folder_path_apparel = "Group4_Project_Data/models/NB_apparel"
model_gcs_path_apparel = f"gs://{bucket_name}/{model_folder_path_apparel}"
nb_apparel.write().overwrite().save(model_gcs_path_apparel)

model_folder_path_automotive = "Group4_Project_Data/models/NB_automotive"
model_gcs_path_automotive = f"gs://{bucket_name}/{model_folder_path_automotive}"
nb_automotive.write().overwrite().save(model_gcs_path_automotive)

model_folder_path_beauty = "Group4_Project_Data/models/NB_beauty"
model_gcs_path_beauty = f"gs://{bucket_name}/{model_folder_path_beauty}"
nb_beauty.write().overwrite().save(model_gcs_path_beauty)

model_folder_path_electronics = "Group4_Project_Data/models/NB_electronics"
model_gcs_path_electronics = f"gs://{bucket_name}/{model_folder_path_electronics}"
nb_electronics.write().overwrite().save(model_gcs_path_electronics)

model_folder_path_shoes = "Group4_Project_Data/models/NB_shoes"
model_gcs_path_shoes = f"gs://{bucket_name}/{model_folder_path_shoes}"
nb_shoes.write().overwrite().save(model_gcs_path_shoes)

model_folder_path_pet_products = "Group4_Project_Data/models/NB_pet_products"
model_gcs_path_pet_products = f"gs://{bucket_name}/{model_folder_path_pet_products}"
nb_pet_products.write().overwrite().save(model_gcs_path_pet_products)

model_folder_path_sports = "Group4_Project_Data/models/NB_sports"
model_gcs_path_sports = f"gs://{bucket_name}/{model_folder_path_sports}"
nb_sports.write().overwrite().save(model_gcs_path_sports)

model_folder_path_toys = "Group4_Project_Data/models/NB_toys"
model_gcs_path_toys = f"gs://{bucket_name}/{model_folder_path_toys}"
nb_toys.write().overwrite().save(model_gcs_path_toys)


# ### Load models from GCP Storage

# In[8]:


model_path_apparel = 'gs://msca-bdp-student-gcs/Group4_Project_Data/models/NB_apparel'
nb_apparel = LogisticRegressionModel.load(model_path_apparel)

model_path_automotive = 'gs://msca-bdp-student-gcs/Group4_Project_Data/models/NB_automotive'
nb_automotive = LogisticRegressionModel.load(model_path_automotive)

model_path_beauty = 'gs://msca-bdp-student-gcs/Group4_Project_Data/models/NB_beauty'
nb_beauty = LogisticRegressionModel.load(model_path_beauty)

model_path_electronics = 'gs://msca-bdp-student-gcs/Group4_Project_Data/models/NB_electronics'
nb_electronics = LogisticRegressionModel.load(model_path_electronics)

model_path_shoes = 'gs://msca-bdp-student-gcs/Group4_Project_Data/models/NB_shoes'
nb_shoes = LogisticRegressionModel.load(model_path_shoes)

model_path_pet_products = 'gs://msca-bdp-student-gcs/Group4_Project_Data/models/NB_pet_products'
nb_pet_products = LogisticRegressionModel.load(model_path_pet_products)

model_path_sports = 'gs://msca-bdp-student-gcs/Group4_Project_Data/models/NB_sports'
nb_sports = LogisticRegressionModel.load(model_path_sports)

model_path_toys = 'gs://msca-bdp-student-gcs/Group4_Project_Data/models/NB_toys'
nb_toys = LogisticRegressionModel.load(model_path_toys)


# ### Compare individual model performances

# In[ ]:


schema = StructType([
    StructField("model", StringType(), True),
    StructField("dataset", StringType(), True),
    StructField("accuracy", DoubleType(), True)
])

data = [("NB Apparel", "Apparel", nb_accuracy_apparel), 
        ("NB Automotive", "Automotive", nb_accuracy_automotive), 
        ("NB Beauty", "Beauty", nb_accuracy_beauty),
        ("NB Electronics", "Electronics", nb_accuracy_electronics),
        ("NB Shoes", "Shoes", nb_accuracy_shoes),
        ("NB Pet Products", "Pet Products", nb_accuracy_pet_products),
        ("NB Sports", "Sports", nb_accuracy_sports),
        ("NB Toys", "Toys", nb_accuracy_toys)]

evaluation = spark.createDataFrame(data, schema)
evaluation.show()


# ### Ensemble the models using soft voting

# In[9]:


# Define the string indexer to encode the labels as integers
indexer = StringIndexer(inputCol='label', outputCol='label_idx')
# Define a function to extract the probability of positive class from the logistic regression model
extract_prob = udf(lambda x: float(x[1]), StringType())

test_datasets = [testData_apparel,testData_automotive,testData_beauty,testData_electronics,testData_shoes,testData_pet_products,testData_sports,testData_toys]
dataset_names = ["Apparel", "Automotive", "Beauty", "Electronics", "Shoes", "Pet Products", "Sports", "Toys"]

schema = StructType([
    StructField("dataset", StringType(), True),
    StructField("accuracy", DoubleType(), True)
])
ensemble_df = spark.createDataFrame([], schema)

for j, test_data in enumerate(test_datasets):
    pred1 = nb_apparel.transform(test_data).withColumn('probability1', extract_prob('probability')).select('review_id','sentiment_label','probability1')
    pred2 = nb_automotive.transform(test_data).withColumn('probability2', extract_prob('probability')).select('review_id','probability2')
    pred3 = nb_beauty.transform(test_data).withColumn('probability3', extract_prob('probability')).select('review_id','probability3')
    pred4 = nb_electronics.transform(test_data).withColumn('probability4', extract_prob('probability')).select('review_id','probability4')
    pred5 = nb_shoes.transform(test_data).withColumn('probability5', extract_prob('probability')).select('review_id','probability5')
    pred6 = nb_pet_products.transform(test_data).withColumn('probability6', extract_prob('probability')).select('review_id','probability6')
    pred7 = nb_sports.transform(test_data).withColumn('probability7', extract_prob('probability')).select('review_id','probability7')
    pred8 = nb_toys.transform(test_data).withColumn('probability8', extract_prob('probability')).select('review_id','probability8')
    
    joined_pred = pred1.join(pred2, on=['review_id'], how='inner').join(pred3, on=['review_id'], how='inner').join(pred4, on=['review_id'], how='inner')                 .join(pred5, on=['review_id'], how='inner').join(pred6, on=['review_id'], how='inner').join(pred7, on=['review_id'], how='inner')                 .join(pred8, on=['review_id'], how='inner')

    col_names = [f'probability{i}' for i in (3,4,5)]
    joined_pred = joined_pred.withColumn('avg_probability', sum([col(c) for c in col_names]) / len(col_names))
    joined_pred = joined_pred.withColumn('predicted_label', udf(lambda x: 1.0 if x > 0.5 else 0.0, DoubleType())('avg_probability'))

    evaluator = MulticlassClassificationEvaluator(labelCol='sentiment_label', predictionCol="predicted_label", metricName="accuracy")
    dataset = dataset_names[j]
    voting_accuracy = evaluator.evaluate(joined_pred)
    ensemble_df = ensemble_df.union(spark.createDataFrame([(dataset, voting_accuracy)], schema))
ensemble_df.show()


# ### Evaluate Ensembled Model on unseen dataset

# In[43]:


df_camera = spark.read.format("csv").option("header", "true").                option("delimiter", "\t").load("gs://msca-bdp-student-gcs/Group4_Project_Data/amazon_reviews_us_Camera_v1_00.tsv")
df_camera = df_camera.dropna().withColumn("star_rating",df_camera.star_rating.cast('int')).withColumn('sentiment', when(col('star_rating') <= 3, 'negative').otherwise('positive'))

result_camera = pipeline.fit(df_camera).transform(df_camera)
result_camera = result_camera.select('review_id','product_id','product_title','sentiment_label','idf_features')

trainData_camera, testData_camera = result_camera.randomSplit([0.7,0.3])


df_grocery = spark.read.format("csv").option("header", "true").                option("delimiter", "\t").load("gs://msca-bdp-student-gcs/Group4_Project_Data/amazon_reviews_us_Grocery_v1_00.tsv")
df_grocery = df_grocery.dropna().withColumn("star_rating",df_grocery.star_rating.cast('int')).withColumn('sentiment', when(col('star_rating') <= 3, 'negative').otherwise('positive'))

result_grocery = pipeline.fit(df_grocery).transform(df_grocery)
result_grocery = result_grocery.select('review_id','product_id','product_title','sentiment_label','idf_features')

trainData_grocery, testData_grocery = result_grocery.randomSplit([0.7,0.3])


# Define the string indexer to encode the labels as integers
indexer = StringIndexer(inputCol='label', outputCol='label_idx')
# Define a function to extract the probability of positive class from the logistic regression model
extract_prob = udf(lambda x: float(x[1]), StringType())

test_datasets = [testData_camera,testData_grocery]
dataset_names = ["Camera", "Grocery"]

schema = StructType([
    StructField("dataset", StringType(), True),
    StructField("accuracy", DoubleType(), True)
])
evaluate_df = spark.createDataFrame([], schema)

for j, test_data in enumerate(test_datasets):
    pred1 = nb_apparel.transform(test_data).withColumn('probability1', extract_prob('probability')).select('review_id','sentiment_label','probability1')
    pred2 = nb_automotive.transform(test_data).withColumn('probability2', extract_prob('probability')).select('review_id','probability2')
    pred3 = nb_beauty.transform(test_data).withColumn('probability3', extract_prob('probability')).select('review_id','probability3')
    pred4 = nb_electronics.transform(test_data).withColumn('probability4', extract_prob('probability')).select('review_id','probability4')
    pred5 = nb_shoes.transform(test_data).withColumn('probability5', extract_prob('probability')).select('review_id','probability5')
    pred6 = nb_pet_products.transform(test_data).withColumn('probability6', extract_prob('probability')).select('review_id','probability6')
    pred7 = nb_sports.transform(test_data).withColumn('probability7', extract_prob('probability')).select('review_id','probability7')
    pred8 = nb_toys.transform(test_data).withColumn('probability8', extract_prob('probability')).select('review_id','probability8')
    
    joined_pred = pred1.join(pred2, on=['review_id'], how='inner').join(pred3, on=['review_id'], how='inner').join(pred4, on=['review_id'], how='inner')                 .join(pred5, on=['review_id'], how='inner').join(pred6, on=['review_id'], how='inner').join(pred7, on=['review_id'], how='inner')                 .join(pred8, on=['review_id'], how='inner')

    col_names = [f'probability{i}' for i in range(1, 9)]
    joined_pred = joined_pred.withColumn('avg_probability', sum([col(c) for c in col_names]) / len(col_names))
    joined_pred = joined_pred.withColumn('predicted_label', udf(lambda x: 1.0 if x > 0.5 else 0.0, DoubleType())('avg_probability'))

    evaluator = MulticlassClassificationEvaluator(labelCol='sentiment_label', predictionCol="predicted_label", metricName="accuracy")
    dataset = dataset_names[j]
    voting_accuracy = evaluator.evaluate(joined_pred)
    evaluate_df = evaluate_df.union(spark.createDataFrame([(dataset, voting_accuracy)], schema))
evaluate_df.show()

