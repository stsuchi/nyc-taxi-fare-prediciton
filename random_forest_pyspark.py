from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import DoubleType, IntegerType, StringType
from pyspark.sql.functions import to_timestamp,from_utc_timestamp
from pyspark.sql.functions import udf, struct
import holidays
from pyspark.sql.functions import hour, year, date_format
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
import numpy as np

sc = SparkContext("local", "Simple App")
spark = SparkSession(sc)

schema = StructType([
    StructField("key", StringType()),
    StructField("fare_amount", DoubleType()),
    StructField("pickup_datetime", StringType()),
    StructField("pickup_longitude", DoubleType()),
    StructField("pickup_latitude", DoubleType()),
    StructField("dropoff_longitude", DoubleType()),
    StructField("dropoff_latitude", DoubleType()),
    StructField("passenger_count", IntegerType()),
])

us_holidays = holidays.US()


def load_and_process_data(path='/Users/shirotsuchiya/Documents/kaggle/nyc_taxi/data/train_sample_5000.csv'):
	df = spark.read.csv('data/train_sample_5000.csv',header=True,mode="DROPMALFORMED", schema=schema)

	# remove null values in any field
	df = df.na.drop()
	df = df.filter(df.fare_amount >= 0)

	# specify the ranges of longitude and latitude
	df = df.filter((df.pickup_longitude <= -70) & (df.pickup_longitude >= -80) \
	& (df.dropoff_longitude <= -70) & (df.dropoff_longitude >= -80) \
	& (df.pickup_latitude <= 45) & (df.pickup_latitude >= 39) \
	& (df.dropoff_latitude <= 45) & (df.dropoff_latitude >= 39) )
	

	# passenger count needs to be between 1 and 10
	df = df.filter((df.passenger_count > 0) & (df.passenger_count <= 10))

	# transform string datetime to local datetime
	df = df.withColumn('utc_time',to_timestamp(df.pickup_datetime,"yyyy-MM-dd HH:mm:ss 'UTC'"))
	df = df.withColumn('local_time',from_utc_timestamp(df.utc_time,'America/New_York'))

	# break donw datetime
	df = df.withColumn('local_hour',hour(df.local_time))
	df = df.withColumn('local_year',year(df.local_time))

	df = df.withColumn('pickup_dow',date_format(df.local_time,'E'))

	return df

def compute_distance(lon1,lat1,lon2,lat2):
    return ((lon1-lon2)**2 + (lat1-lat2)**2)**(1/2)

def create_dist_feature(df):
	compute_distance_udf = udf(lambda x: compute_distance(x[0],x[1],x[2],x[3]),DoubleType())
	df = df.withColumn('dist',compute_distance_udf(struct('pickup_longitude','pickup_latitude',\
                                                     'dropoff_longitude','dropoff_latitude')))

	return df

def get_hour(x):
    if x >= 0 and x < 3:
        return 'hour_0'
    elif x >= 3 and x < 6:
        return 'hour_1'
    elif x >= 6 and x < 9:
        return 'hour_2'
    elif x >= 9 and x < 12:
        return 'hour_3'
    elif x >= 12 and x < 15:
        return 'hour_4'
    elif x >= 15 and x < 18:
        return 'hour_5'
    elif x >= 18 and x < 21:
        return 'hour_6'
    elif x >= 21 and x < 24:
        return 'hour_7'
    else:
        return 'hour_8'
    


def get_year(x):
    return 'year_' + str(x)


def get_holiday(x):
	if x in us_holidays:
		return 'holiday'
	else:
		return 'non-holiday'

def create_time_features(df):
	get_hour_udf = udf(lambda x: get_hour(x),StringType())
	get_year_udf = udf(lambda x: get_year(x),StringType())
	get_holiday_udf = udf(lambda x: get_holiday(x),StringType())

	df = df.withColumn('pickup_hour',get_hour_udf(df.local_hour))
	df = df.withColumn('pickup_year',get_year_udf(df.local_year))
	df = df.withColumn('holiday',get_holiday_udf(df.local_time))

	return df

def get_zone(lon,lat):
	zone_number = int((lon+80)*10)
	zone_number += int((lat-39)*10)
	return 'zone_' + str(zone_number)

def create_zone_features(df):
	get_zone_udf = udf(lambda x: get_zone(x[0],x[1]), StringType())

	df = df.withColumn('pickup_zone',get_zone_udf(struct(df.pickup_longitude,df.pickup_latitude)))
	df = df.withColumn('dropoff_zone',get_zone_udf(struct(df.dropoff_longitude,df.dropoff_latitude)))

	return df


def encode_df(df):
	hour_indexer = StringIndexer(inputCol='pickup_hour',outputCol='pickup_hour_numeric')
	year_indexer = StringIndexer(inputCol='pickup_year',outputCol='pickup_year_numeric')
	dow_indexer = StringIndexer(inputCol='pickup_dow',outputCol='pickup_dow_numeric')
	holiday_indexer = StringIndexer(inputCol='holiday',outputCol='holiday_numeric')
	pickup_zone_indexer = StringIndexer(inputCol='pickup_zone',outputCol='pickup_zone_numeric')
	dropoff_zone_indexer = StringIndexer(inputCol='dropoff_zone',outputCol='dropoff_zone_numeric')

	hour_encoder = OneHotEncoder(inputCol='pickup_hour_numeric',outputCol='pickup_hour_vector')
	year_encoder = OneHotEncoder(inputCol='pickup_year_numeric',outputCol='pickup_year_vector')
	dow_encoder = OneHotEncoder(inputCol='pickup_dow_numeric',outputCol='pickup_dow_vector')
	holiday_encoder = OneHotEncoder(inputCol='holiday_numeric',outputCol='holiday_vector')
	pickup_zone_encoder = OneHotEncoder(inputCol='pickup_zone_numeric',outputCol='pickup_zone_vector')
	dropoff_zone_encoder = OneHotEncoder(inputCol='dropoff_zone_numeric',outputCol='dropoff_zone_vector')


	assembler = VectorAssembler(inputCols=['dist','pickup_hour_vector','pickup_year_vector','pickup_dow_vector',\
	                                      'holiday_vector', 'pickup_zone_vector','dropoff_zone_vector','passenger_count'],\
	                           outputCol="features")

	pipeline = Pipeline(stages=[hour_indexer,year_indexer,dow_indexer,holiday_indexer,pickup_zone_indexer,dropoff_zone_indexer,\
	                           hour_encoder,year_encoder,dow_encoder,holiday_encoder,pickup_zone_encoder,dropoff_zone_encoder,\
	                           assembler])

	transformed = pipeline.fit(df).transform(df)

	return transformed

def train_test_split(transformed,train_fraction=0.8):
	(trainingData, testData) = transformed.randomSplit([train_fraction, 1 - train_fraction])
	
	return trainingData, testData


def train_and_predict_with_rf(trainingData,testData):
	rf = RandomForestRegressor(featuresCol='features', labelCol='fare_amount', numTrees=400, maxDepth=10, seed=42)
	model = rf.fit(trainingData)

	predictions = model.transform(testData)

	return predictions


def compute_rmse(predictions):
	evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='fare_amount', metricName='rmse')
	rmse = evaluator.evaluate(predictions)
	return rmse
	

def main():
	df = load_and_process_data()
	df = create_dist_feature(df)
	df = create_time_features(df)
	df = create_zone_features(df)
	data = encode_df(df)
	trainingData,testData = train_test_split(data)
	predictions = train_and_predict_with_rf(trainingData,testData)
	#predictions.show(5)
	rmse = compute_rmse(predictions)
	print('Test RMSE: ', rmse)
	


if __name__=='__main__':
	main()