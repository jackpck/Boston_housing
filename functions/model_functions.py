import sys
sys.path.append('../')

from nlp_functions import preprocess
import requests
import json
import pandas as pd
from pandas.io.json import json_normalize
from math import sqrt,radians,cos,sin,asin
from geopy.geocoders import Nominatim
import pickle
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class RegrSwitcher(BaseEstimator):
    def __init__(self,estimator=RandomForestRegressor()):
        self.estimator = estimator

    def fit(self,x,y=None,**kwargs):
        self.estimator.fit(x,y)
        return self

    def predict(self,x,y=None):
        return self.estimator.predict(x)

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
        geolocator = Nominatim(timeout=3)
        location = geolocator.geocode(address_str)
        return location.latitude, location.longitude


    def get_poi(self, address, poi):

        lat_prop, long_prop = self.get_coor(address)
        overpass_query = """
        [out:json]; 
        (node['{0}'='{1}']({2},{3},{4},{5}););
         out center;
        """.format(Model.POI_KEY[poi],poi,
                   Model.BOSTON_S,
                   Model.BOSTON_W,
                   Model.BOSTON_N,
                   Model.BOSTON_E)
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
                       property_type,remarks):

        N_poi = self.count_poi(address)

        x_input = np.empty(14,dtype=object)
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
        x_input[12] = property_type
        x_input[13] = has_lot

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



    M = Model()
    path_to_model = '../pickled_models/RF_all_property_sold_price.pkl'
    M.load_from_pickle(path_to_model)

    X_input = M.to_input_array(year,sqft,bed,bath,
                               lot_size,has_lot,address,
                               property_type,remarks)
    print('estimated sold price: ',M.predict(X_input))
