from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

import pandas as pd
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
import Common_Module.Common_Module as CM

retailDF = pd.read_excel(io='D:\\OnlineRetail.xlsx')
print(retailDF.info())
retailDF = retailDF[retailDF['Quantity']>0]
retailDF = retailDF[retailDF['UnitPrice']>0]
retailDF = retailDF[retailDF['CustomerID'].notnull()]
print(retailDF.shape)
print(retailDF.isnull().sum())
print(retailDF['Country'].value_counts()[:5])
retailDF = retailDF[retailDF['Country']=='United Kingdom']
print(retailDF.shape)

#
retailDF['sale_amount'] = retailDF['Quantity'] * retailDF['UnitPrice']
retailDF['CustomerID'] = retailDF['CustomerID'].astype(int)
print(retailDF['CustomerID'].value_counts().head())
print(retailDF.groupby('CustomerID')['sale_amount'].sum().sort_values(ascending=False)[:5])

aggregations = {
    'InvoiceDate': 'max',
    'InvoiceNo': 'count',
    'sale_amount': 'sum'
}
cust_df = retailDF.groupby('CustomerID').agg(aggregations)
cust_df = cust_df.rename(columns={'InvoiceDate':'Recency', 'InvoiceNo':'Frequency', 'sale_amount':'Monetary'})
# 컬럼명 바꾼다
cust_df = cust_df.reset_index()
cust_df['Recency'] = datetime.datetime(2021,10,2) - cust_df['Recency']
cust_df['Recency'] = cust_df['Recency'].apply(lambda x:x.days+1)
print(cust_df)