import pandas as pd
import numpy as np
import us
import re
from sklearn.preprocessing import LabelEncoder


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

def decimalize_feet(s:str) -> float:
    """Converts feet'inch" to decimal feet"""
    res = np.nan
    if type(s) == str:
        m = re.match(r'^(\d+)\'(\d+)\"$', s.replace(" ", ""))
        if m:
            res = float(m.group(1)) + float(m.group(2)) / 12.
    return res

def encode_labels(df:pd.DataFrame, labels:list) -> dict:
    """Normalize labels in dataframe such that they contain only values between 0 and n_labels-1
    
    Returns:
        Dict of labels with their respective encoders for decoding if needed
    """
    # Create Encode Dict
    encode_dict = {}
    # Copy DataFrame
    df_copy = df.copy()
    for label in labels:
        # Create Label Encoding Object
        label_encode = LabelEncoder()
        # Fit Label Encoder
        label_encode.fit(df[label].unique().tolist())
        # Transform Feature
        encoded = label_encode.transform(df[label])
        df_copy[label] = encoded
        # Append to Dict
        encode_dict[label] = label_encode
        
    return df_copy, encode_dict

def decode_labels(df:pd.DataFrame, label_encoders:dict) -> None:
    """Reverse transformed labels in dataframe such that they contain original values"""
    # Loop through encoders to reassign values back to dictionary
    for label, encoder in label_encoders.items():
        # Inverse Transform Feature
        decoded = encoder.inverse_transform(df[label])
        # Reassign DF Labels
        df[label] = decoded