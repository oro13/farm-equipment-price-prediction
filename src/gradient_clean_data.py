import pandas as pd
import numpy as np
import us
import re
from src.helpers import *

def gradient_clean_df(df:pd.DataFrame) -> tuple:
    df_cop = df.copy()
    # Drop Columns
    df_cop.drop(columns=['MachineID', 
                    ], inplace=True)
    
    # Create "Vehicle Type" Feature from "fiProductClassDesc"
    df_cop["Vehicle Type"] = df_cop["ProductGroupDesc"]
    df_cop["Power Rating"] = df_cop["fiProductClassDesc"].apply(lambda x:(x.partition("-")[-1]))
    df_cop.drop(columns='fiProductClassDesc', inplace=True)
    
    # Make a horsepower column
    df_hp = df_cop[df_cop["Power Rating"].str.contains('horsepower', case=False)]
    df_cop['HorsePower'] = df_hp['Power Rating'].map(getReMax)
    
    # Convert state names to state numbers
    us_dict = us.states.mapping('name', 'fips')
    us_dict["Washington DC"] = 3
    print(us_dict)
    us_dict["Unspecified"] = '0'
    df_cop["state"]=df_cop["state"].map(lambda x: int(us_dict[x.strip()]))
    
    
    return df