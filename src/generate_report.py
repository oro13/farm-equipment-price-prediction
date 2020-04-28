#!/usr/bin/env python3

# Import Libraries
import pandas as pd
from pandas_profiling import ProfileReport

success = [
    "\nCreated dataframe from data/churn.csv ...\n",
    "\nCreating profiling object ...\n",
    "\nCreating HTML file \"churn_report.html\" ...\n",
    "\nAll done! Check your directory to see the HTML report, open it using a web browser\n",
    
    
]

error = """
Error occured, check csv file path in "generate_report.py" 
"""


if __name__ == '__main__':
    try:
        # Create Dataframe
        print(success[0])
        df = pd.read_csv("../data/Train.csv")
        
        # Generate Profile
        print(success[1])
        profile = ProfileReport(df, title='Profiling Report', html={'style':{'full_width':True}})
        
        # Output Profile as HTML in Local Dir
        print(success[2])
        profile.to_file(output_file="profile_report.html")

        # Validate Success
        print(success[3])
        
    except:
        print(error)