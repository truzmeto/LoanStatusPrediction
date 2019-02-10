import numpy as np
import pandas as pd

def CleanData(df):
    """
    Function to performe data cleaning with pandas
    """
    
    #rm '%' and convert to float
    df.int_rate = pd.Series(df.int_rate).str.replace('%', '').astype(float)
    df.revol_util = pd.Series(df.revol_util).str.replace('%', '').astype(float)
    
    df.replace('n/a', np.nan,inplace = True)
    df.emp_length.fillna(value = 0,inplace = True)
    
    df['emp_length'].replace(to_replace = '[^0-9]+', value='',
                             inplace = True, regex = True)
    df['emp_length'] = df['emp_length'].astype(int)
    
    issue_month = df.issue_d.str.replace('-2015', '')
    df['issue_month'] = pd.Series( issue_month, index=df.index)
    
    df.term = pd.Series(df.term).str.replace('months', '').astype(int)
    
    df.earliest_cr_line = pd.to_datetime(df.earliest_cr_line)
    df.issue_d = pd.to_datetime(df.issue_d)
    
    cred_age = df.issue_d - df.earliest_cr_line
    df['cred_age'] = pd.Series( cred_age, index=df.index)
        
    tmp = np.rint(df['cred_age'].map(lambda x: x.days/365))
    df['cred_age'] = pd.Series( tmp, index = df.index)

    df.drop(columns="issue_d",inplace = True)
    df.drop(columns="earliest_cr_line",inplace = True)
    
    return df    


def ColorList(colors):
    """
    Function to return specified color by index.
    -------------------------------------------
    Input:
    colors -- list of color indecies( 1,2,3...)
    
    Output:
    cols   -- list with color code
    """
    color_list = ["#9d6d00", "#903ee0", "#11dc79", "#f568ff", "#419500", "#013fb0",
                  "#f2b64c", "#007ae4", "#ff905a", "#33d3e3", "#9e003a", "#019085",
                  "#950065", "#afc98f", "#ff9bfa", "#83221d", "#01668a", "#ff7c7c",
                  "#643561", "#75608a"]

    cols = []
    for i in range(len(colors)):
        indx = colors[i]
        cols.append(color_list[indx])

    return cols
    
