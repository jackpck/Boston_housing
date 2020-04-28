import pandas as pd
import numpy as np

property_type = 'single_family_residential'
poi_list = ['convenience','supermarket','park','school','station','stop_position']
radius = 0.5

df_all = pd.DataFrame()
for poi in poi_list:
    df_poi = pd.read_csv('./data/geodata/processed/' +
                         '%s_%s_radius_%.2f.csv'%(poi,property_type,radius),index_col=0)
    df_all = pd.concat([df_all,df_poi],axis=1)

df_all.columns = poi_list

df_all.to_csv('./data/processed/' + 'Boston_%s_poi.csv'%property_type)
