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


# Create the application object
app = Flask(__name__)
app.config['GOOGLEMAPS_KEY'] = apikey
GoogleMaps(app)

#@app.route('/',methods=["GET","POST"]) #we are now using these methods to get user input
#def home_page():
#    return render_template(html_script)  # render a template

lat_default = 42.36012
lng_default = -71.0589

@app.route('/',methods=["GET","POST"])
def enter_address():
    if request.method == 'POST':
        address_input = request.values.get('address') + ', Boston, MA'

        print(address_input)
        property_latitude,property_longitude = get_coor(address_input)

        #if address_input == "":
        #        print('address input is empty')
        #        return render_template(html_script,
        #                               my_form_result="Empty")
        #else:

        print('creating a map in the view')
        print(property_latitude,property_longitude)

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

