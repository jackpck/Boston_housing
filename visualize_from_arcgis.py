'''
Getting poi from arcgis, plot coordinates on Google Maps
'''
import pandas as pd
import numpy as np
import gmplot as gmplot
from arcgis.gis import GIS
from arcgis.geocoding import geocode
import ast
import matplotlib.pyplot as plt

import seaborn as sns

df_prop = pd.read_csv('./data/' + 'redfin_2020-01-19-08-17-26.csv')
df_poi = pd.read_json('./data/' + 'education.json')

filter_key = 'amenity'
filter_value = 'school'

mask = df_poi['tags'].apply(lambda x: True if filter_key in x and x[filter_key] == filter_value else False)
df_poi = df_poi[mask]


gmap = gmplot.GoogleMapPlotter(df_prop.LATITUDE.iloc[0],
                               df_prop.LONGITUDE.iloc[0], 13)

with open('./data/key/' + 'google_apikey.txt') as FILE:
    gmap.apikey = FILE.readline().rstrip()

#x_coor, y_coor = zip(*[(df_poi.lat.iloc[i],df_poi.lon.iloc[i]) for i in range(len(df_poi))])

#gis = GIS(profile='dd')
#pdx_map = gis.map('Portland, OR')
#pdx_map.basemap='gray'
#pdx_map
#plt.show()

gmap.scatter(df_poi.lat.values, df_poi.lon.values, '#3B0B39', size=40, marker=False)
gmap.draw("./maps_html/%s_map.html"%'school')

