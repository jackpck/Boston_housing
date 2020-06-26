import sys
sys.path.append('./functions')

from flask import Flask, render_template, request, Response, redirect
import pandas as pd
import numpy as np
import pygal
from flask_googlemaps import GoogleMaps
from flask_googlemaps import Map
from geopy.geocoders import Nominatim
from model_functions import Model
from io import BytesIO


html_script = 'product-page.html'

with open('./data/key/' + 'google_apikey.txt') as FILE:
    apikey = FILE.readline().rstrip()


# Load model
Agent = Model()
path_to_model = './pickled_models/RF_all_property_sold_price.pkl'
Agent.load_from_pickle(path_to_model)


# Create the application object
app = Flask(__name__)
app.config['GOOGLEMAPS_KEY'] = apikey
GoogleMaps(app)

lat_default = 42.36012
lng_default = -71.0589

property_type_dict = {'Single'}


@app.route('/',methods=["GET","POST"])
def enter_address():
    if request.method == 'POST':
        address = request.values.get('address') + ', Boston, MA'
        year_built = int(request.values.get('year_built'))
        sqft = float(request.values.get('sqft'))
        beds = int(request.values.get('beds'))
        baths = int(request.values.get('baths'))
        lot_size = float(request.values.get('lot_size'))
        property_type = request.values.get('property_type')
        remark = request.values.get('remark')

        if lot_size > 0:
            has_lot = 1
        else:
            has_lot = 0

        property_latitude,property_longitude = Agent.get_coor(address)
        X_input = Agent.to_input_array(year_built, sqft, beds, baths, lot_size,
                                       has_lot, address, property_type, remark)
        sold_price_pred = Agent.predict(X_input)

        gmap = Map(
            identifier="view-side",
            lat=property_latitude,
            lng=property_longitude,
            markers=[(property_latitude, property_longitude)],
            style="height:500px;width:600px;"
        )

        return render_template(html_script, gmap=gmap,
                               address=address,
                               year_built=year_built,
                               sqft=sqft,
                               beds=beds,
                               baths=baths,
                               lot_size=lot_size,
                               property_type=property_type,
                               sold_price_pred='%s'%'${:,.0f}'.format(float(sold_price_pred)))

    else:
        gmap = Map(
            identifier="view-side",
            lat=lat_default,
            lng=lng_default,
            markers=[(lat_default,lng_default)],
            style = "height:500px;width:600px;"
        )

        return render_template(html_script, gmap=gmap)



# start the server with the 'run()' method
if __name__ == "__main__":
    app.run(debug=True,port=8000)  # will run locally http://127.0.0.1:5000/

