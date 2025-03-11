import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

# 1) Load dataset and EDA
data = pd.read_csv("mentalhealth.csv")
print(data.head(10))

print("\nDescriptive statistics of dataset:\n", data.describe())

print("\nBasic info on the dataset:")
data.info()

print("\nNumber of records and variables:\n", data.shape)

print("\nNumber of missing values in each variable:\n", data.isnull().sum())

# 2) Clean dataset
clean_data = data.drop(data.columns[0], axis=1) #Drop variable "Unnamed: 0"
print(clean_data.head(10))

## Replace missing values in 'statement' variable with 'Únknown'
clean_data.fillna({"statement": "Unknown"}, inplace=True)
print("\nNumber of missing values in each variable (clean_data):\n",
      clean_data.isnull().sum())

## Check for duplicated records
num_duplicated = clean_data.duplicated().sum()
print("\nNumber of duplicated records:\n", num_duplicated)

## Handle duplicated records
clean_data.drop_duplicates(inplace=True) #Remove duplicated records
print("\nNumber of duplicated records (after dropping them):\n",
      clean_data.duplicated().sum())

## Number of categories of status variable
status_count = clean_data['status'].value_counts(ascending=True)
print("\nNumber of categories of status variable:\n", status_count)

## Visualise 'status' categories
plt.figure(figsize=(10, 5))
status_count.plot(kind='bar', color='blue', edgecolor='black')

plt.xlabel("Mental Health Categories")
plt.ylabel("Number of Records")
plt.title("Number of Records per Mental Health Category")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

## Format 'status' categories into binary classification
clean_data["status"] = clean_data["status"].apply(
    lambda x: "Normal" if x == "Normal" else "Depression")

status_counts = clean_data['status'].value_counts(ascending=True)
print("\nNumber of categories of status variable (after formatting):\n", status_counts)

## Handling missing records in 'status' variable
clean_data["status"] = clean_data["status"].fillna(clean_data["status"].mode()[0])
print("\nNumber of missing values in 'status' variable (after imputation):\n", clean_data.isnull().sum())

# 3) Encode 'status' into binary (0 or 1)
from sklearn.preprocessing import LabelEncoder
le_status = LabelEncoder()
clean_data['status_encoded'] = le_status.fit_transform(clean_data['status'])
clean_data = clean_data.drop(columns=['status'], axis=1)

## Mapping between encoded values & original status
label_mapping = dict(zip(le_status.classes_, le_status.transform(le_status.classes_)))
for status, encoded in label_mapping.items():
    print(f"\nStatus: {status}, Encoded Value: {encoded}") #Print mapping

print("\nclean_data DataFrame after 'status' variable encoded:\n", clean_data.head(10))

# 4) Export dataset
clean_data.to_csv("mentalhealth_clean.csv", index=False)