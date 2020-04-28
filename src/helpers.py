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