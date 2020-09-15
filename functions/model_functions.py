import sys
sys.path.append('../')

from nlp_functions import preprocess
import requests
import json
import csv
import pandas as pd
from datetime import datetime
from pandas.io.json import json_normalize
from math import sqrt,radians,cos,sin,asin
from geopy.geocoders import Nominatim
import pickle
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
import numpy as np


def get_latest_data_via_API():
    '''
    Does not work. Need to somehow bypass usage analysis algorithms
    '''
    CSV_URL = "https://www.redfin.com/stingray/api/gis-csv?al=1&market=boston&min_stories=1&num_homes=100&ord=redfin-recommended-asc&page_number=1&region_id=1826"+\
              "&region_type=6&sold_within_days=90&status=9&uipt=1,2,3,4,5,6&v=8"
    response = requests.get(CSV_URL)
    return response.content


def load_data():
    property_types = ['single_family_residential', 'condo', 'townhouse']

    df = pd.DataFrame([])
    for property_type in property_types:
        df_temp = pd.read_csv('./data/raw_joined/' + 'Boston_%s_joined_dataframe.csv' % property_type, index_col=0)
        df_temp['PROPERTY TYPE'] = property_type
        df = pd.concat([df, df_temp])
    df = df.sort_values('LIST DATE')

    df['DAYS ON MKT'] = df['DAYS ON MKT'].apply(lambda x: x if x > 0 else np.nan)
    df['PREMIUM'] = (df['SOLD PRICE'] - df['LIST PRICE']) / df['LIST PRICE']
    df['PREMIUM'] = df['PREMIUM'].apply(lambda x: x if np.abs(x) < 2 else np.nan)
    df['LIST MONTH'] = pd.to_datetime(df['LIST DATE']).apply(lambda x: x.month)
    df['HAS LOT'] = df['LOT SIZE'].apply(lambda x: 1 if x > 0 else 0)
    df.pop('LIST DATE')
    df.pop('SOLD DATE')
    df.pop('HOA/MONTH')
    df = df.dropna()

    return df


class RegrSwitcher(BaseEstimator):
    def __init__(self,estimator=RandomForestRegressor()):
        self.estimator = estimator

    def fit(self,x,y=None,**kwargs):
        self.estimator.fit(x,y)
        return self

    def predict(self,x,y=None):
        return self.estimator.predict(x)

    def predict_proba(self,x,y=None):
        return self.estimator.predict_proba(x)

    def score(self,x,y):
        return self.estimator.score(x,y)


class Model:
    FEATURES = ['YEAR BUILT', 'SQUARE FEET', 'BEDS',
                'BATHS', 'LOT SIZE', 'REMARKS',
                'convenience', 'supermarket', 'park',
                'school', 'station', 'stop_position',
                'PROPERTY TYPE','HAS LOT']
    POI_KEY = {'station':'railway','stop_position':'public_transport','school':'amenity',
               'convenience':'shop','supermarket':'shop','park':'leisure'}

    OVERPASS_URL = "http://overpass-api.de/api/interpreter"
    BOSTON_S = 42.25
    BOSTON_W = -71.15
    BOSTON_N = 42.4
    BOSTON_E = -70.95

    RADIUS = 0.5  # km
    EARTH_RADIUS = radiusEarth = 6371 #km

    NUM_FEATURES = ['SQUARE FEET', 'YEAR BUILT']  # require BoxCox transformation
    TXT_FEATURES = ['REMARKS']  # require preprocessing
    CAT_FEATURES = ['HAS LOT', 'PROPERTY TYPE']  # require one hot encoding


    def __init__(self):
        self.model = None


    def load_from_pickle(self,path):
        self.model = pickle.load(open(path, 'rb'))


    def haversine_dist(self,lat1, lat2, lon1, lon2):
        lat1 = radians(lat1)
        lat2 = radians(lat2)
        lon1 = radians(lon1)
        lon2 = radians(lon2)
        return 2 * Model.EARTH_RADIUS * asin(
            sqrt((sin(lat1 - lat2) / 2) ** 2 + cos(lat1) * cos(lat2) * (sin(lon1 - lon2) / 2) ** 2))


    def delta_lat(self):
        return Model.RADIUS/Model.EARTH_RADIUS


    def delta_lon(self,lat):
        return np.abs(2*asin(sin(Model.RADIUS/(2*Model.EARTH_RADIUS))/cos(lat)))


    def get_poi_within_radius(self,lat1,lat2,long1,long2,r):
        n = len(lat2)
        poi_coor = []
        for i in range(n):
            if self.haversine_dist(lat1, lat2[i], long1, long2[i]) < r:
                poi_coor.append((lat2[i],long2[i]))
        return poi_coor


    def get_coor(self, address_str):
        '''
        Get latitude and longitude of an address
        e.g. 25 Brighton Street, Boston, MA
        '''
        geolocator = Nominatim(timeout=3,user_agent='boston_housing')
        location = geolocator.geocode(address_str)
        return location.latitude, location.longitude


    def get_poi(self, address, poi):
        lat_prop, long_prop = self.get_coor(address)
        south_boundary = lat_prop - self.delta_lat()
        west_boundary = long_prop - self.delta_lon(lat_prop)
        north_boundary = lat_prop + self.delta_lat()
        east_boundary = long_prop + self.delta_lon(lat_prop)

        overpass_query = """
        [out:json]; 
        (node['{0}'='{1}']({2},{3},{4},{5}););
         out center;
        """.format(Model.POI_KEY[poi],poi,
                   south_boundary,
                   west_boundary,
                   north_boundary,
                   east_boundary)
        response = requests.get(Model.OVERPASS_URL,
                                params={'data': overpass_query})
        df = pd.DataFrame.from_dict(json_normalize(response.json()))

        poi_lats = [x['lat'] for x in df['elements'][0]]
        poi_lons = [x['lon'] for x in df['elements'][0]]
        poi_coor = self.get_poi_within_radius(lat_prop,poi_lats,long_prop,poi_lons,Model.RADIUS)
        return poi_coor


    def count_poi(self, address):
        return {poi:len(self.get_poi(address, poi)) for poi in Model.POI_KEY}


    def to_input_array(self,year,sqft,bed,bath,
                       lot_size,has_lot,address,
                       property_type,remarks,est_ppsqft):

        N_poi = self.count_poi(address)

        x_input = np.empty(16,dtype=object)
        x_input[0] = year
        x_input[1] = sqft
        x_input[2] = bed
        x_input[3] = bath
        x_input[4] = lot_size
        x_input[5] = preprocess(remarks)
        x_input[6] = N_poi['convenience']
        x_input[7] = N_poi['supermarket']
        x_input[8] = N_poi['park']
        x_input[9] = N_poi['school']
        x_input[10] = N_poi['station']
        x_input[11] = N_poi['stop_position']
        x_input[12] = est_ppsqft
        x_input[13] = property_type
        x_input[14] = datetime.today().month
        x_input[15] = has_lot

        return x_input.reshape(1,-1)


    def predict(self,x):
        return 10**self.model.predict(x)[0] # model is trained with log(y)


if __name__ == '__main__':
    import numpy as np

    year = 2000
    sqft = 1000
    bed = 3
    bath = 2
    lot_size = 100
    has_lot = 1
    address = '25 Brighton Ave, Boston, MA'
    property_type = 'condo'
    remarks = 'This condo has a beautiful renovated kitchen. New AC as well.'
    remarks = 'This condo is average. Nothing special. Kinda boring.'
    est_ppsqft = 400 # user input est $/sqft. Try to automate this part.



    M = Model()
    path_to_model = '../pickled_models/RF_all_property_sold_price.pkl'
    M.load_from_pickle(path_to_model)

    X_input = M.to_input_array(year,sqft,bed,bath,
                               lot_size,has_lot,address,
                               property_type,remarks,est_ppsqft)
    print('estimated sold price: ',M.predict(X_input))
