import numpy as np
import pandas as pd
from dateutil import tz, parser
import holidays
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

us_holidays = holidays.US()

def preporcess_data(path='/Users/shirotsuchiya/Documents/kaggle/nyc_taxi/data/train.csv',nrows=5000):
	data = pd.read_csv(path,nrows=nrows)

	print('null values: ')
	print(data.isnull().sum())

	data = data.dropna()

	# remove negative fare_amount
	data = data[data.fare_amount >= 0]

	# narrow down longitude and latitude
	data = data[(data.pickup_longitude >= -80) & (data.pickup_longitude <= -70) & \
	(data.pickup_latitude >= 39) & (data.pickup_latitude <= 50) ]

	data = data[(data.dropoff_longitude >= -80) & (data.dropoff_longitude <= -70) & \
	(data.dropoff_latitude >= 39) & (data.dropoff_latitude <= 50) ]

	return data

def create_distance_features(data):
	data['long_diff'] = np.abs(data.pickup_longitude  - data.dropoff_longitude)
	data['lat_diff'] = np.abs(data.pickup_latitude - data.dropoff_latitude)
	data['dist'] = ((data.long_diff)**2 + (data.lat_diff)**2)**(1/2)

	return data

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

def get_holidays(x):
	if x in us_holidays:
		return 'holiday'
	else:
		return 'non-holiday'


def create_date_features(data):
	to_zone = tz.gettz('America/New_York')
	data['pickup_datetime'] = data.pickup_datetime.apply(lambda x: parser.parse(x))
	data['local_time'] = data.pickup_datetime.apply(lambda x: x.astimezone(to_zone))
	data['local_hour'] = data.local_time.apply(lambda x: x.hour)
	data['local_year'] = data.local_time.apply(lambda x: x.year)
	data['pickup_hour'] = data.local_hour.apply(get_hour)
	data['pickup_year'] = data.local_year.apply(get_year)
	data['pickup_dow'] = data.local_time.apply(lambda x: x.strftime('%A'))
	data['holiday'] = data.local_time.apply(lambda x: get_holidays(x))

	return data

def get_zone(lon,lat):
	zone_number = int((lon+80)*10)
	zone_number += int((lat-39)*10)

	return 'zone_' + str(zone_number)


def create_zone_features(data):
	data['pickup_zone'] = data.apply(lambda row: get_zone(row['pickup_longitude'], row['pickup_latitude']), axis=1)
	data['dropoff_zone'] = data.apply(lambda row: get_zone(row['dropoff_longitude'],row['dropoff_latitude']),axis=1)

	return data


def transform_features(data):
	X = np.column_stack((data.dist,pd.get_dummies(data.pickup_hour),pd.get_dummies(data.pickup_year),pd.get_dummies(data.pickup_dow), \
		pd.get_dummies(data.holiday),pd.get_dummies(data.pickup_zone),pd.get_dummies(data.dropoff_zone), data.passenger_count))

	Y = data.fare_amount.values

	return X, Y

def train_random_forest(X,Y,n_estimators=300,max_depth=10):
	regr = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth, random_state=0)
	regr.fit(X,Y)

	return regr

def predict_with_random_forest(X,regr):
	predictions = regr.predict(X)

	return predictions

def compute_rmse(predictions, Y):
	rmse = (np.sum([(i-j)**2 for i,j in zip(predictions,Y)])/float(len(predictions)))**(1/2)
	return rmse

def perform_grid_search(X,Y):
	parameters = {'max_depth':[5,10,20,30,40,50], 'n_estimators':[250,300,350,400,450,500]}
	regr = RandomForestRegressor(random_state=0)
	model = GridSearchCV(regr,parameters)
	model.fit(X,Y)
	print(model.best_params_)

def main():
	data = preporcess_data()
	data = create_distance_features(data)
	data = create_date_features(data)
	#print(data[['pickup_hour','pickup_year','pickup_dow','holiday']].head(5))
	data = create_zone_features(data)
	
	X, Y = transform_features(data)
	train_X, test_X, train_Y, test_Y = train_test_split(X,Y,test_size=0.20, random_state=42)
	
	perform_grid_search(X,Y)
	'''
	regr = train_random_forest(train_X,train_Y)
	predictions = predict_with_random_forest(test_X,regr)
	rmse = compute_rmse(predictions,test_Y)
	print(rmse)
	'''

if __name__=='__main__':
	main()