# Importing Library

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Loading Data

df=pd.read_csv("Data\\train.csv")

# Exploring Data

print(df.head(10))

print("\nTo check null values:")
print(df.isnull().sum())

print("\nTo check datatypes:")
print(df.dtypes)

# Data Preprocessing

#Checking correlation of features containing null values to determine how they should be handled
df_numerics_only = df.select_dtypes(include=np.number)
sns.heatmap(df_numerics_only.corr(),annot=True)

#As both features other than the target variable show high positive correlation... instead of imputing them with mean,
#rows with null values will be dropped to get accurate predictions with model
df.dropna(subset=["Resource Allocation"], inplace=True)
df.dropna(subset=["Mental Fatigue Score"], inplace=True)
df.dropna(subset=["Burn Rate"], inplace=True)

#Converting Date of Joining to datetime datatype
df['Date of Joining'] = pd.to_datetime(df['Date of Joining'])

#Feature engineering: Days at Company
df["Days at Company"]=(pd.Timestamp('today') - df['Date of Joining']).dt.days

#Label Encoding Categorical Columns
label_e=LabelEncoder()
df["Gender"]=label_e.fit_transform(df["Gender"]) #F=0, M=1
df["Company Type"]=label_e.fit_transform(df["Company Type"]) #Service=1, Product=0
df["WFH Setup Available"]=label_e.fit_transform(df["WFH Setup Available"]) #No=0, Yes=1

#Drop irrelevant columns
df.drop(columns=['Employee ID', 'Date of Joining'], inplace=True)

# Saving Cleaned Data

df.to_csv("D:\\Atomcamp\\Data\\Cleaned_ML_Dataset.csv",index=False)