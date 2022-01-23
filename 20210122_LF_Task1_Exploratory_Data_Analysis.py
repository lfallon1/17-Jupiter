# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 14:25:39 2022

@author:     LF
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

#Please insert path to python file in below string
working_directory = r"..."
os.chdir(working_directory)

# =========================================================================================================
# Functions
# =========================================================================================================

def remove_empty_rows_and_cols(df):
    df.dropna(how='all', inplace=True)
    df.reset_index(drop=True,inplace=True)
    for field in list(df):    
        if df[field].isnull().values.all()==True: # empty so delete column
            df.drop(columns=[field],inplace=True)
            
    return df


def get_col_widths(dataframe):
    idx_max = max([len(str(s)) for s in dataframe.index.values] + [len(str(dataframe.index.name))])
    return [idx_max] + [max([len(str(s)) for s in dataframe[col].values] + [len(col)]) for col in dataframe.columns]

def _format_spreadsheet(tab,dataframe):
    dataframe.reset_index(drop=True)
    # Removed for deliverable two
    dataframe.index+=1
    dataframe.to_excel(writer,sheet_name = tab,index = False)
    worksheet = writer.sheets[tab]
    workbook.add_format({'text_wrap':True})
    for i, width in enumerate(get_col_widths(dataframe)):
        if width >100:
            width = 100
        else:
            width = 20
        worksheet.set_column(i,i,width)
# =========================================================================================================
# Params
# =========================================================================================================

todays_date = datetime.today().strftime('%Y%m%d')
timestamp = datetime.today().strftime('%Y%m%d-%H%M%S')
output_path = os.path.join(working_directory,r"Exploratory Data Analysis")
os.chdir(output_path)
filename = todays_date + r'_anglo_american_analysis.xlsx'
writer = pd.ExcelWriter(os.path.join(output_path,filename),engine = 'xlsxwriter')
workbook = writer.book


# ingest Anglo American platinum results as dictionary of dataframes
anglo_2021 = pd.read_excel(os.path.join(working_directory,r"Exploratory Data Analysis\anglo-american-country-by-country-reporting-data-2020.xlsx"),sheet_name='Table 1 Revenue', header = 4)
anglo_2020 = pd.read_excel(os.path.join(working_directory,r"Exploratory Data Analysis\anglo-american-country-by-country-reporting-data-2019.xlsm"),sheet_name='Table 1 Revenue', header = 4)
anglo_2019 = pd.read_excel(os.path.join(working_directory,r"Exploratory Data Analysis\anglo-american-country-by-country-reporting-data-2018-01.xlsm"),sheet_name='Table 1 Revenue', header = 4)

# =========================================================================================================
# 2021
# =========================================================================================================
# check column headers 
list(anglo_2021) # columns 1, 4 and 5 are not coming through due to merged cells. Also there are line breaks in some columns.  
anglo_2021.iloc[0] 
# looking at the next row we can see that Revenues is a combination of Unrelated Party, Related Party and Total. 
# Still nothing in first column so checking the entirity of the first column to see what it includes

anglo_2021.rename(columns={'Revenues':'Revenues - Unrelated Party','Unnamed: 3': 'Revenues - Related Party','Unnamed: 4': 'Revenues - Total',
                           'CBCR Effective Tax Rate\n%': 'CBCR Effective Tax Rate',
                           'Statutory\nCorporate\nTax Rate\n%': 'Statutory Corporate Tax Rate',
                           'Income Tax Accrued (Current Year)': 'Income Tax Accrued - Current Year' #after initial run I found this column is different to other years
                           },inplace=True)

# use remove empty rows and cols function 
anglo_2021 = remove_empty_rows_and_cols(anglo_2021)

# opening the data frame in the variable explorer I can see that the first country under jurisdiction 
# doesn't start util row 7. The rows before are financials and might be useful at a later date for QC (quality control)

anglo_2021_sum = anglo_2021.iloc[:4].reset_index(drop=True)

anglo_2021 = anglo_2021.iloc[4:].reset_index(drop=True)

anglo_2021['Tax Jurisdiction'].unique() # still have NULL values so checking what these contain

df_null_tax = anglo_2021[anglo_2021['Tax Jurisdiction'].isnull()].reset_index(drop=True)
for field in list(df_null_tax):
    print(str(field)+':',df_null_tax[field].isnull().values.all()) 

# last column contains values relating to the country above it, going to save the country and description
# columns as a separate DF in case we need them later
anglo_2021_explanation = anglo_2021[['Tax Jurisdiction','Explanation of significant differences in the rates']]

anglo_2021.drop(columns='Explanation of significant differences in the rates',inplace=True)
# now remove the null rows
anglo_2021.dropna(how='all', inplace=True)
anglo_2021['Tax Jurisdiction'].isnull().values.any() #no null values in country column now
anglo_2021 = anglo_2021[~anglo_2021['Tax Jurisdiction'].isnull()].reset_index(drop=True) #the jurisdiction will be a mandatory field in analysis. 

for field in list(anglo_2021):
    print(str(field)+':',len(anglo_2021[field].unique()),anglo_2021[field].dtype)


# =========================================================================================================
# 2020
# =========================================================================================================
anglo_2020 = remove_empty_rows_and_cols(anglo_2020)
list(anglo_2020)
# using similar logic to previous section to clean data
anglo_2020.iloc[0] 
anglo_2020.rename(columns={'Revenues':'Revenues - Unrelated Party','Unnamed: 2': 'Revenues - Related Party','Unnamed: 3': 'Revenues - Total'
                           },inplace=True)  

anglo_2020.drop([0],inplace=True) #remove first row
anglo_2020.reset_index(drop=True,inplace=True)
for field in list(anglo_2020):
    print(str(field)+':',anglo_2020[field].isnull().values.all(),len(anglo_2020[field].unique()),anglo_2020[field].dtype)

#field Unnamed: 12 only has two values plus Note has 4, check what they are and remove if unimportant
anglo_2020['Unnamed: 12'].unique()
anglo_2020['Note'].unique()
anglo_2020.drop(columns=['Unnamed: 12','Note'],inplace=True)
anglo_2020.reset_index(drop=True,inplace=True)
anglo_2020 = anglo_2020[~anglo_2020['Tax Jurisdiction'].isnull()].reset_index(drop=True)



# =========================================================================================================
# 2019
# =========================================================================================================
# this section uses the same logic as above to produce a cleansed dataset
anglo_2019 = remove_empty_rows_and_cols(anglo_2019)
list(anglo_2019)
anglo_2019.iloc[0] 
anglo_2019.rename(columns={'Revenues':'Revenues - Unrelated Party','Unnamed: 2': 'Revenues - Related Party','Unnamed: 3': 'Revenues - Total',
                           'Income Tax Paid (on Cash basis)\n':  'Income Tax Paid (on Cash basis)',
                           'Income Tax Accrued - Current Year\n': 'Income Tax Accrued - Current Year',
                           'Stated Capital\n': 'Stated Capital',
                           'Accumulated Earnings\n': 'Accumulated Earnings',
                           'Number of employees\n': 'Number of employees',
                           'Tangible Assets other than Cash and Cash equivalents\n(Mandatory)': 'Tangible Assets other than Cash and Cash equivalents'
                           
                           },inplace=True)  
anglo_2019.drop([0],inplace=True) #remove first row
anglo_2019['Unnamed: 12'].unique()
anglo_2019['Note'].unique()
anglo_2019.drop(columns=['Unnamed: 12','Note'],inplace=True)
anglo_2019.reset_index(drop=True,inplace=True)
anglo_2019 = anglo_2019[~anglo_2019['Tax Jurisdiction'].isnull()].reset_index(drop=True)
for field in list(anglo_2020):
    print(str(field)+':',anglo_2020[field].isnull().values.all(),len(anglo_2020[field].unique()),anglo_2020[field].dtype)

# =========================================================================================================
# Combine and export
# =========================================================================================================

# Concatenate the three dataframes and export for us to begin analysis
anglo_2019['Year'] = 2019
anglo_2020['Year'] = 2020
anglo_2021['Year'] = 2021

anglo_df = pd.concat([anglo_2019,anglo_2020,anglo_2021])

# Exporting to excel spreadsheet as I will be producing a report in Power Point 
_format_spreadsheet( 'Anglo American Revenue '+todays_date,anglo_df)
writer.save()

# =========================================================================================================
