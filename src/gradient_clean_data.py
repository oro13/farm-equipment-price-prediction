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
    
    "TO ENCODE"
    
    df_cop["Turbocharged"] = df_cop["Turbocharged"].map(lambda x: "Unspecified" if x != "Yes" else x)
    df_cop["UsageBand"] = df_cop["UsageBand"].map(lambda x: "Unspecified" if pd.isnull(x) else x)
    df_cop["Horsepower_Type"] = df_cop["Engine_Horsepower"].map(lambda x: "Unspecified" if pd.isnull(x) else x)
    df_cop["Drive_System"] = df_cop["Drive_System"].map(lambda x: "Unspecified" if pd.isnull(x) else x)
    df_cop["Enclosure"] = df_cop["Enclosure"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Forks"] = df_cop["Forks"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Pad_Type"] = df_cop["Pad_Type"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Ride_Control"] = df_cop["Ride_Control"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Stick"] = df_cop["Stick"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Transmission"] = df_cop["Transmission"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Blade_Extension"] = df_cop["Blade_Extension"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Blade_Width"] = df_cop["Blade_Width"].map(lambda x: x[:-1] if (type(x) == str and x != "None or Unspecified") else "Unspecified")
    df_cop["Enclosure_Type"] = df_cop["Enclosure_Type"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Hydraulics"] = df_cop["Hydraulics"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Pushblock"] = df_cop["Pushblock"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Scarifier"] = df_cop["Scarifier"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Tip_Control"] = df_cop["Tip_Control"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Coupler"] = df_cop["Coupler"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Coupler_System"] = df_cop["Coupler_System"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Hydraulics_Flow"] = df_cop["Hydraulics_Flow"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Track_Type"] = df_cop["Track_Type"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Thumb"] = df_cop["Thumb"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Pattern_Changer"] = df_cop["Pattern_Changer"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Grouser_Type"] = df_cop["Grouser_Type"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Backhoe_Mounting"] = df_cop["Backhoe_Mounting"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Blade_Type"] = df_cop["Backhoe_Mounting"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Travel_Controls"] = df_cop["Travel_Controls"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Differential_Type"] = df_cop["Differential_Type"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Steering_Controls"] = df_cop["Steering_Controls"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["fiSecondaryDesc"] = df_cop["fiSecondaryDesc"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "#NAME?") else x.strip())
    df_cop["ProductSize"] = df_cop["ProductSize"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Ripper"] = df_cop["Ripper"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Grouser_Tracks"] = df_cop["Grouser_Tracks"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    
    
    'NON ENCODE'
    df_cop["Tire_Size"] = pd.to_numeric(df_cop["Tire_Size"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else getReMax(x) if (x == "10 inch") else x), errors='coerce', downcast='float')
    df_cop["Undercarriage_Pad_Width"] = pd.to_numeric(df_cop["Undercarriage_Pad_Width"].map(lambda x: np.nan if (x == "None or Unspecified") else getReMax(x) if (x != "31.5 inch" and pd.notnull(x)) else 31.5), errors='coerce', downcast='float')
    df_cop["Stick_Length"] = df_cop["Stick_Length"].map(lambda x: np.nan if (x == "None or Unspecified") else decimalize_feet(x))
    
    
    'CONSISTENCY'
    
    # Modify Enclosure Feature for Consistency
    # EROPS w AC == EROPS AC, NO ROPS == Unspecified
    sub = {"EROPS w AC":"EROPS AC", "NO ROPS":"Unspecified"}
    df_cop["Enclosure"] = df_cop["Enclosure"].map(lambda x: sub[x] if (x in sub) else x)
    # Modify Transmission Feature for Consistency
    # Autoshift == AutoShift, 
    sub = {"AutoShift":"Autoshift"}
    df_cop["Transmission"] = df_cop["Transmission"].map(lambda x: sub[x] if (x in sub) else x)
    
    
    # Convert SalesDate to datetime object
    df_cop["saledate"] = pd.to_datetime(df_cop["saledate"])

    
#     'ENCODE COLUMNS'
#     le = LabelEncoder()
#     # List of features to encode
#     features = ["UsageBand", "fiBaseModel", "fiSecondaryDesc", "fiModelDescriptor", "ProductSize", "state", "Vehicle Type", ]
    
    'DROP COLUMNS'

    # Drop Unecessary Columns
    df_cop.drop(columns=['MachineID', 'SalesID', 'auctioneerID', 'fiModelDescriptor', 
                    ], inplace=True)
    
    # Drop Dupe Columns
    df_cop.drop(columns =['fiProductClassDesc', 'ProductGroupDesc', 'ProductGroup', 'fiModelDesc', 'fiModelSeries', 'Engine_Horsepower'], inplace=True)
    # Drop ModelDesc | Keep Base Model
    
    'RENAME COLUMNS FOR CLARITY'
#     df_cop = df_cop.rename(columns={'fiModelDesc': 'Model Description'})
    
    # Return train/target split
    X = df_cop
    y = df_cop.pop("SalePrice")
    
    return X, y