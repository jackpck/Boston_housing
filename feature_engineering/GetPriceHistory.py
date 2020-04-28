'''
Premium: difference between the closing price and the original listing price
historical price format: "['$500,000','-','-','$490,000',...]"
convert historical price to nice list of float
find price_sold-price_list (latest)
date of selling and listing should not be the same
Next: scrap the description of each event from Redfin
'''
import pandas as pd
import numpy as np
import sys
import re
sys.path.append('~/PycharmProjects/Boston_housing/')

property_type = 'condo'

df_price = pd.read_csv('./data/' + 'Boston_%s_price_history.csv'%property_type,index_col=0)
n_prop = len(df_price.index)
indices = df_price.index.tolist()

sold_date = []
list_date = []
sold_price = []
list_price = []

index = []

for i in range(n_prop):
    price = df_price.iloc[i]['HISTORICAL PRICE']
    price = price.replace(chr(8212),'$-1') # ASCII code of '-' in string is 8212 (not equal to ord('-'))
    price = price.replace(',','')
    price_list = re.findall("(?<=\$)(.*?)(?=')",price)
    price_list = [int(p) for p in price_list]

    date = df_price.iloc[i]['HISTORICAL DATE']
    date = date.replace(",","")
    date_list = re.findall("(?<=')(\w.*?)(?=')",date)

    description = df_price.iloc[i]['HISTORICAL DESCRIPTION']
    description = description.replace(",","")
    description_list = re.findall("(?<=')(\w.*?)(?=')",description)

    try:
        sold_idx = min([i for i in range(len(description_list)) if 'Sold' in description_list[i]])
        list_idx = min([i for i in range(len(description_list)) if 'Listed' in description_list[i]])

        list_price.append(price_list[list_idx])
        sold_price.append(price_list[sold_idx])
        list_date.append(date_list[list_idx])
        sold_date.append(date_list[sold_idx])

        index.append(df_price.index[i])

    except ValueError:
        pass


df = pd.DataFrame(np.array([list_price,sold_price,list_date,sold_date]).T,
                  columns=['LIST PRICE','SOLD PRICE','LIST DATE','SOLD DATE'],
                  index=index)

df.to_csv('./data/' + 'Boston_%s_transaction.csv'%property_type)


