from flask import Flask, render_template, request, Response, redirect
import pandas as pd
import numpy as np
import pygal
from flask_googlemaps import GoogleMaps
from flask_googlemaps import Map
from geopy.geocoders import Nominatim
from io import BytesIO

def get_coor(address_str):
    '''
    Get latitude and longitude of an address
    e.g. 25 Brighton Street, Boston, MA
    '''
    geolocator = Nominatim(timeout=3)
    location = geolocator.geocode(address_str)
    return location.latitude, location.longitude

html_script = 'product-page.html'

with open('./data/key/' + 'google_apikey.txt') as FILE:
    apikey = FILE.readline().rstrip()


df_condo = pd.read_csv('./data/processed/Boston_condo_feature_matrix.csv')
df_sfr = pd.read_csv('./data/processed/Boston_single_family_residential_feature_matrix.csv')
df_townhouse = pd.read_csv('./data/processed/Boston_townhouse_feature_matrix.csv')


# Create the application object
app = Flask(__name__)
app.config['GOOGLEMAPS_KEY'] = apikey
GoogleMaps(app)


lat_default = 42.36012
lng_default = -71.0589

@app.route('/',methods=["GET","POST"])
def enter_address():
    if request.method == 'POST':
        address_input = request.values.get('address') + ', Boston, MA'
        property_type = request.args.get('property type')

        property_latitude,property_longitude = get_coor(address_input)


        gmap = Map(
            identifier="view-side",
            lat=property_latitude,
            lng=property_longitude,
            markers=[(property_latitude, property_longitude)]
        )

        return render_template(html_script, gmap=gmap)

    else:
        gmap = Map(
            identifier="view-side",
            lat=lat_default,
            lng=lng_default,
            markers=[(lat_default,lng_default)]
        )

        return render_template(html_script, gmap=gmap)



# start the server with the 'run()' method
if __name__ == "__main__":
    app.run(debug=True,port=8000)  # will run locally http://127.0.0.1:5000/

