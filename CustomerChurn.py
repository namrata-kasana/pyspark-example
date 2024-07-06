# libraries
import warnings
# import findspark
import pandas as pd
import seaborn as sns
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
from pyspark.ml.feature import Bucketizer

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

#Creating A Spark Session
spark = SparkSession.builder.master("local[*]").getOrCreate()

#🔍Exploratory Data Analysis
spark_df = spark.read.csv("./Resources/churn2.csv",inferSchema=True,header=True)
# spark_df.show(10)

#no of records
# print("Shape",spark_df.count(),len(spark_df.columns))

#types of Variables
# spark_df.printSchema()

#summary statistics
# spark_df.show(5)

#summary statistics for specific variables
# spark_df.describe(["age","exited"]).show()

#class statistics of categorical variables
# spark_df.groupBy("exited").count().show()

# unique classes
# spark_df.select("exited").distinct().show()

# groupby transactions
# spark_df.groupby("exited").count().show()

#time spent
# spark_df.groupby("exited").agg({"tenure":"mean"}).show()

#Selection and summary statistics of all numeric variables
# num_cols = [col[0] for col in spark_df.dtypes if col[1] != 'string']
# spark_df.select(num_cols).describe().show()

#Selection and summary statistics of all categorical variables
# num_cols = [col[0] for col in spark_df.dtypes if col[1] == 'string']
# spark_df.select(num_cols).describe().show()

# mean of numerical variables relative to the target variable
# for col in [col.lower() for col in num_cols]:
#     spark_df.groupby("exited").agg({col:"mean"}).show()

# ✍🏼Data Preprocessing & Feature Engineering


