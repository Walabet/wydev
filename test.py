import pandas as pd 
import numpy as np
dataset = pd.read_csv('C:\\Users\\User\\Desktop\\PFE\\diabetes_1_cleaned.csv')
dataset
dataset.shape
dataset.info()
dataset.describe()
dataset.isnull().sum()
print(dataset.columns)
X = dataset.drop('Outcome',axis=1)
y = dataset['Outcome']
print(X)
print(y)

print(dataset['Pregnancies'].value_counts())
print(dataset['Glucose'].value_counts())
print(dataset['BloodPressure'].value_counts())
print(dataset['SkinThickness'].value_counts())
print(dataset['Insulin'].value_counts())
print(dataset['BMI'].value_counts())
print(dataset['DiabetesPedigreeFunction'].value_counts())
print(dataset['Age'].value_counts())
