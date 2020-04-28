import pandas as pd
import numpy as np
import us
import re

def set_ohe(df:pd.DataFrame, col_name:str):
    for val in df[col_name].value_counts().index:
        df[f"{col_name}: {val}"] = df[col_name].map(lambda x: 1.0 if x==val else 0.0 )
    df.drop(df.columns[-1], axis=1, inplace=True)
    return df
        
def getReMax(val:str) -> np.float:
    """Returns maximum number in a string using regex"""
    search = re.findall('\d+', val) 
    nums = map(np.float, search) 
    return max(nums)

def clean_df(df):
    
    #drop some columns
    df.drop(columns=['UsageBand','Blade_Extension', 'Blade_Width', 'Enclosure_Type',
                     'Engine_Horsepower', 'Pushblock', 'Scarifier', 'Tip_Control',
                     'Coupler_System', 'Grouser_Tracks', 'Hydraulics_Flow','Backhoe_Mounting', 
                     'Blade_Type', 'Travel_Controls','Differential_Type','Steering_Controls',
                     'SalesID','fiBaseModel','fiSecondaryDesc',
                     'fiModelSeries','fiModelDescriptor', 'auctioneerID',
                     'Drive_System', 'datasource'
                    ], inplace=True)
    
    # Find the logs of target values, SalePrice
    
    try:
        df['SalePrice'] = np.log(df['SalePrice'])
    except:
        pass
    
    # Convert MachineHoursCurrent Meter "NaN" values to the average value
    df['MachineHoursCurrentMeter'].fillna(df['MachineHoursCurrentMeter'].mean(), inplace=True)
    
    # Clean Ripper Values
    df = set_ohe(df, "Ripper")
    df.drop(columns='Ripper', inplace=True)
    
    # Clean ProductSize Values
    df = set_ohe(df, "ProductSize")
    df.drop(columns='ProductSize', inplace=True)

    
    # Clean YearMade Values - Change Year 1000 to Average Year
    df['YearMade'].replace(1000, np.NaN, inplace=True)
    df['YearMade'].fillna(df['YearMade'].mean(), inplace=True)
    
    # Clean fiProductClassDesc
    # Create "Vehicle Type" Feature from "fiProductClassDesc"
    df["Vehicle Type"] = df["ProductGroupDesc"]
    df["Power Rating"] = df["fiProductClassDesc"].apply(lambda x:(x.partition("-")[-1]))
    df.drop(columns='fiProductClassDesc', inplace=True)
    
    # Make a horsepower column
    df_hp = df[df["Power Rating"].str.contains('horsepower', case=False)]
    df['HorsePower'] = df_hp['Power Rating'].map(getReMax)
    df['HorsePower'].fillna(df['HorsePower'].mean(), inplace=True)
    
    set_ohe(df, "Vehicle Type")
    df.drop(columns='Vehicle Type', inplace=True)
    df.drop(columns='Power Rating', inplace=True)
    df.drop(columns='ProductGroupDesc', inplace=True)
    
    # Get year of sale only from saledate
    df['saledate'] = pd.to_datetime(df['saledate'])
    df['yearsold'] = df['saledate'].map(lambda x: x.year)
    df.drop(columns='saledate', inplace=True)
    
    # Convert state names to state numbers
    us_dict = us.states.mapping('name', 'fips')
    us_dict["Washington DC"] = us_dict.pop("District of Columbia")
    us_dict["Unspecified"] = '0'
    df["state"]=df["state"].map(lambda x: int(us_dict[x.strip()]))
    
    
    # Drop some more columns
    df.drop(columns=['ProductGroup', 'Enclosure','Pad_Type', 'fiModelDesc','Forks',
                     'Ride_Control','Stick','Transmission','Turbocharged','Hydraulics',
                     'Tire_Size','Coupler','Track_Type', 'Undercarriage_Pad_Width',
                     'Stick_Length','Thumb','Pattern_Changer','Grouser_Type', 'MachineID', 'ModelID'
                    ], inplace=True)
    
    return df