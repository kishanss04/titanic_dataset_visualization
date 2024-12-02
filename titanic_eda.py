import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('train.csv')

print("Dataset Shape:", df.shape)
print(df.head())
print(df.info())

missing_data = df.isnull().sum()
print("\nMissing Values:\n", missing_data)

sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

df.drop(columns=['Cabin'], inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

print("\nMissing Values After Cleaning:\n", df.isnull().sum())

sns.histplot(df['Age'], kde=True, bins=30, color='blue')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.show()

sns.countplot(x='Survived', data=df, palette='pastel')
plt.title("Survival Count (0 = Did not survive, 1 = Survived)")
plt.show()

sns.barplot(x='Sex', y='Survived', data=df, palette='viridis')
plt.title("Survival Rate by Gender")
plt.show()

sns.barplot(x='Pclass', y='Survived', data=df, palette='magma')
plt.title("Survival Rate by Passenger Class")
plt.show()

correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

print("\nInsights:")
print("- Females have a higher survival rate than males.")
print("- Passengers in higher classes (1st) have a better survival chance.")
print("- Age distribution is skewed, and most passengers are between 20-40 years old.")
