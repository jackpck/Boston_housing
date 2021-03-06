'''
json file from overpass turbo
'''

import pandas as pd
import numpy as np
from math import sqrt,radians,cos,sin,asin
import matplotlib.pyplot as plt

radiusEarth = 6371 #km

def haversine_dist(lat1,lat2,lon1,lon2):
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    return 2*radiusEarth*asin(sqrt((sin(lat1-lat2)/2)**2 + cos(lat1)*cos(lat2)*(sin(lon1-lon2)/2)**2))

def count_number_within_radius(df,lat2,long2,r):
    n = len(lat2)
    count = 0
    for i in range(n):
        if haversine_dist(df.LATITUDE,lat2.iloc[i],df.LONGITUDE,long2.iloc[i]) < r:
            count += 1
    return count


property_type = 'condo'
poi = 'park_garden'
filter_key = 'leisure'
filter_value = 'park'
radius = 0.5 # km

df_prop = pd.read_csv('./data/raw/' + 'redfin_2020-01-19-08-17-26.csv')
df_history = pd.read_csv('./data/processed/' + 'Boston_%s_transaction.csv'%property_type,index_col=0)

df_prop.pop('SOLD DATE')
df = df_prop.join(df_history, how = 'inner')

print('number of %s: %i'%(property_type,len(df)))

df_poi = pd.read_json('./data/geodata/raw/' + '%s.json'%poi)

# filter tags:
mask = df_poi['tags'].apply(lambda x: True if filter_key in x and x[filter_key] == filter_value else False)
print('total number of key: ',len(df_poi))
df_poi = df_poi[mask]
print('number of key with value=%s: %i'%(filter_value,len(df_poi)))

print('counting')
df['count'] = df.apply(lambda x:count_number_within_radius(x,df_poi.lat,df_poi.lon,radius),axis = 1)

print('total number of %s: %i'%(filter_key,len(df_poi)))
df['count'].to_frame().to_csv('./data/geodata/processed/' + '%s_%s_radius_%.2f.csv'%(filter_value,property_type,radius))

#df['count'].hist(bins=max(df['count']))
#plt.show()


