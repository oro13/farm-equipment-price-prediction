import pandas as pd
import numpy as np
import us
import re
from sklearn.preprocessing import LabelEncoder
from src.helpers import *

def gradient_clean_Xy(df:pd.DataFrame) -> tuple:
    # Copy Dataframe
    df_cop = df.copy()
    
    
    'CREATE/CONVERT COLUMNS'
    
    # Create "Vehicle Type" Feature from "fiProductClassDesc"
    df_cop["Vehicle_Type"] = df_cop["ProductGroupDesc"]
    df_cop["Power_Rating"] = df_cop["fiProductClassDesc"].apply(lambda x:(x.partition("-")[-1]))
    
    # Make a horsepower column
    df_hp = df_cop[df_cop["Power_Rating"].str.contains('horsepower', case=False)]
    df_cop['Horsepower'] = df_hp['Power_Rating'].map(getReMax)
    
    # Convert state names to state numbers
    us_dict = us.states.mapping('name', 'abbr')
    us_dict["Washington DC"] = 'DC'
    us_dict["Unspecified"] = 'None'
    df_cop["state"] = df_cop["state"].map(lambda x: us_dict[x.strip()])
    
    # Convert "None or Unspecified rows into Unspecified"
    df_cop["Turbocharged"] = df_cop["Turbocharged"].map(lambda x: "Unspecified" if x != "Yes" else x)
    # Convert NaN rows into Unspecified"
    df_cop["UsageBand"] = df_cop["UsageBand"].map(lambda x: "Unspecified" if pd.isnull(x) else x)
    df_cop["Engine_Horsepower"] = df_cop["Engine_Horsepower"].map(lambda x: "Unspecified" if pd.isnull(x) else x)
    df_cop["Drive_System"] = df_cop["Drive_System"].map(lambda x: "Unspecified" if pd.isnull(x) else x)
    df_cop["Enclosure"] = df_cop["Enclosure"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Forks"] = df_cop["Forks"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Pad_Type"] = df_cop["Pad_Type"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Ride_Control"] = df_cop["Ride_Control"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    
    
    # Modify Enclosure Feature for Consistency
    # EROPS w AC == EROPS AC, NO ROPS == Unspecified
    sub = {"EROPS w AC":"EROPS AC", "NO ROPS":"Unspecified"}
    df_cop["Enclosure"] = df_cop["Enclosure"].map(lambda x: sub[x] if (x in sub) else x)
    
    
    
    
    # Convert SalesDate to datetime object
    df_cop["saledate"] = pd.to_datetime(df_cop["saledate"])

    
    'ENCODE COLUMNS'
    le = LabelEncoder()
    # List of features to encode
    features = ["UsageBand", "fiBaseModel", "fiSecondaryDesc", "fiModelDescriptor", "ProductSize", "state", "Vehicle Type", ]
    
    'DROP COLUMNS'

    # Drop Unecessary Columns
    df_cop.drop(columns=['MachineID', 'SalesID', 'auctioneerID', 
                    ], inplace=True)
    
    # Drop Dupe Columns
    df_cop.drop(columns =['fiProductClassDesc', 'ProductGroupDesc', 'ProductGroup', 'fiModelDesc', 'fiModelSeries'], inplace=True)
    # Drop ModelDesc | Keep Base Model
    
    'RENAME COLUMNS FOR CLARITY'
#     df_cop = df_cop.rename(columns={'fiModelDesc': 'Model Description'})
    
    # Return train/target split
    X = df_cop
    y = df_cop.pop("SalePrice")
    
    return X, y