import pandas as pd 
import numpy as np 
import re 
from sklearn.preprocessing import LabelEncoder

def getReMax(val:str) -> np.float:
    """Returns maximum number in a string using regex"""
    search = re.findall('\d+', val) 
    nums = map(np.float, search) 
    return max(nums)

def decimalize_feet(s:str) -> float:
    """Converts feet'inch" to decimal feet"""
    res = np.nan
    if type(s) == str:
        m = re.match(r'^(\d+)\'(\d+)\"$', s.replace(" ", ""))
        if m:
            res = float(m.group(1)) + float(m.group(2)) / 12.

    return res

def set_ohe(df:pd.DataFrame, col_name:str):
    for val in df[col_name].value_counts().index:
        df[f"{col_name}: {val}"] = df[col_name].map(lambda x: 1.0 if x==val else 0.0 )
    df.drop(df.columns[-1], axis=1, inplace=True)
    return df


def clean_this_df(df:pd.DataFrame) -> pd.DataFrame:
    '''
    inputs
    -------
    df: pandas dataframe with a total of 53 columns

    returns
    ------
    X: feature data, numpy array
    y: target data, saleprice, numpy array
    '''

    # Copy Dataframe
    df_cop = df.copy()
    

    # Create "Vehicle Type" Feature from "ProductGroupDesc"
    df_cop['Vehicle_Type'] = df_cop['ProductGroupDesc']
    df_cop['Power_Rating'] = df_cop['fiProductClassDesc'].apply(lambda x: (x.partition('-')[-1]))

    # Create a "Horsepower" column
    df_hp = df_cop[df_cop['Power_Rating'].str.contains('horsepower', case=False)]
    df_cop['Horsepower'] = df_hp['Power_Rating'].map(getReMax)
    # Fill "Horsepower" column NaN values with 0. 
    df_cop['Horsepower'].fillna(0, inplace=True)

    # Create a "Capacity_Lbs" column
    df_capacity = df_cop[df_cop['Power_Rating'].str.contains('operating', case=False)]
    df_cop['Capacity_Lbs'] = df_capacity['Power_Rating'].map(getReMax)
    # Fill "Capacity_Lbs" column NaN values with 0. 
    df_cop['Capacity_Lbs'].fillna(0, inplace=True)

    # Create a "Capacity_Metric_Tons"
    df_capacity = df_cop[df_cop['Power_Rating'].str.contains('metric', case=False)]
    df_cop['Capacity_Metric_Tons'] = df_capacity['Power_Rating'].map(getReMax)
    # Fill "Capacity_Metric_Tons" column NaN values with 0. 
    df_cop['Capacity_Metric_Tons'].fillna(0, inplace=True)

    # To Encode
    df_cop["UsageBand"] = df_cop["UsageBand"].map(lambda x: "Unspecified" if pd.isnull(x) else x)
    df_cop["Enclosure"] = df_cop["Enclosure"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Ride_Control"] = df_cop["Ride_Control"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Hydraulics"] = df_cop["Hydraulics"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Hydraulics_Flow"] = df_cop["Hydraulics_Flow"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Track_Type"] = df_cop["Track_Type"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Grouser_Type"] = df_cop["Grouser_Type"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["ProductSize"] = df_cop["ProductSize"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)
    df_cop["Ripper"] = df_cop["Ripper"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else x)

    # Non Encode
    df_cop["Tire_Size"] = pd.to_numeric(df_cop["Tire_Size"].map(lambda x: "Unspecified" if (pd.isnull(x) or x == "None or Unspecified") else getReMax(x) if (x == "10 inch") else x), errors='coerce', downcast='float')
    df_cop["Stick_Length"] = df_cop["Stick_Length"].map(lambda x: np.nan if (x == "None or Unspecified") else decimalize_feet(x))
    ### Make YearMade median year if the year is 1000.
    median_year_made = (df_cop['YearMade'].replace(1000, np.NaN)).mean()
    df_cop["YearMade"] = df_cop["YearMade"].map(lambda x: median_year_made if x == 1000 else x)

    # Consistency
    ### Modify Enclosure Feature for Consistency
    ### EROPS w AC == EROPS AC, NO ROPS == Unspecified
    sub = {"EROPS w AC":"EROPS AC", "NO ROPS":"Unspecified"}
    df_cop["Enclosure"] = df_cop["Enclosure"].map(lambda x: sub[x] if (x in sub) else x)

    # Convert SalesDate to datetime object
    df_cop["saledate"] = pd.to_datetime(df_cop["saledate"])
    df_cop["yearsold"] = df_cop["saledate"].map(lambda x: x.year)

    # Replace remaining NaN with 0.
    df_cop.fillna(0, inplace=True)

    # Drop Columns That Won't Be Used
    columns_to_drop = ['SalesID', 'MachineID','ModelID','datasource','auctioneerID','saledate',
                  'fiModelDesc','fiBaseModel', 'fiSecondaryDesc', 'fiModelSeries',
                  'fiModelDescriptor','fiProductClassDesc','state',
                  'ProductGroup','ProductGroupDesc','Drive_System','Forks','Pad_Type',
                  'Stick','Transmission','Turbocharged','Blade_Extension',
                  'Blade_Width', 'Enclosure_Type','Engine_Horsepower',
                  'Pushblock','Scarifier','Tip_Control','Coupler','Coupler_System',
                  'Grouser_Tracks', 'Undercarriage_Pad_Width', 'Blade_Type', 'Travel_Controls',
                  'Differential_Type','Steering_Controls', 'Power_Rating', 'Thumb',
                  'Pattern_Changer', 'Backhoe_Mounting']
    df_cop.drop(columns=columns_to_drop, inplace=True)
    
    # One Hot Encode Label Columns
    to_one_hot_columns = ['UsageBand', 'ProductSize', 'Enclosure','Ride_Control','Hydraulics',
                    'Hydraulics_Flow','Ripper','Track_Type','Grouser_Type', 'Vehicle_Type']

    for each_column in to_one_hot_columns:
        df_cop = set_ohe(df_cop, each_column)
        df_cop.drop(columns=each_column,inplace=True)

    return df_cop


