import pandas as pd
import numpy as np
import us
import re
from src.helpers import *

def gradient_clean_df(df:pd.DataFrame) -> tuple:
    
#     # Drop Columns
#     df.drop(columns=['UsageBand','Blade_Extension', 'Blade_Width', 'Enclosure_Type',
#                      'Engine_Horsepower', 'Pushblock', 'Scarifier', 'Tip_Control',
#                      'Coupler_System', 'Grouser_Tracks', 'Hydraulics_Flow','Backhoe_Mounting', 
#                      'Blade_Type', 'Travel_Controls','Differential_Type','Steering_Controls',
#                      'SalesID','fiBaseModel','fiSecondaryDesc',
#                      'fiModelSeries','fiModelDescriptor', 'auctioneerID',
#                      'Drive_System', 'datasource'
#                     ], inplace=True)
    
    # Create "Vehicle Type" Feature from "fiProductClassDesc"
    df["Vehicle Type"] = df["ProductGroupDesc"]
    df["Power Rating"] = df["fiProductClassDesc"].apply(lambda x:(x.partition("-")[-1]))
    df.drop(columns='fiProductClassDesc', inplace=True)
    
    # Make a horsepower column
    df_hp = df[df["Power Rating"].str.contains('horsepower', case=False)]
    df['HorsePower'] = df_hp['Power Rating'].map(getReMax)
    
    return df