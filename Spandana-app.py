import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Title
st.title("Automobile Dataset Preprocessing - Exam Part 1")
st.markdown("""
#### Rejeti Spandana Sharon  
**Banner ID**: 001409139
""")

# Load Data
st.header("Load and Display Data")
url = "https://raw.githubusercontent.com/klamsal/Fall2024Exam/main/auto.csv"
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
           "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
           "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
           "peak-rpm","city-mpg","highway-mpg","price"]
df = pd.read_csv(url, names=headers)

# Replace '?' with NaN
df.replace("?", np.nan, inplace=True)
st.write("Initial Data Preview:", df.head())

# Identify Missing Values
st.subheader("Missing Value Summary")
missing_data = df.isnull()
missing_counts = df.isnull().sum()
st.dataframe(missing_counts[missing_counts > 0])

# Replace with Mean
df["normalized-losses"] = df["normalized-losses"].astype("float")
df["normalized-losses"].fillna(df["normalized-losses"].mean(), inplace=True)
df["bore"] = df["bore"].astype("float")
df["bore"].fillna(df["bore"].mean(), inplace=True)
df["stroke"] = df["stroke"].astype("float")
df["stroke"].fillna(df["stroke"].mean(), inplace=True)
df["horsepower"] = df["horsepower"].astype("float")
df["horsepower"].fillna(df["horsepower"].mean(), inplace=True)
df["peak-rpm"] = df["peak-rpm"].astype("float")
df["peak-rpm"].fillna(df["peak-rpm"].mean(), inplace=True)

# Replace with Most Frequent
df["num-of-doors"].fillna(df["num-of-doors"].value_counts().idxmax(), inplace=True)

# Drop rows where price is NaN
df.dropna(subset=["price"], inplace=True)
df.reset_index(drop=True, inplace=True)

# Fix data types
df["normalized-losses"] = df["normalized-losses"].astype("int")
df["price"] = df["price"].astype("float")
df["peak-rpm"] = df["peak-rpm"].astype("float")

# Data Standardization
st.subheader("Fuel Consumption Transformation")
df['city-L/100km'] = 235 / df['city-mpg'].astype(float)
df['highway-L/100km'] = 235 / df['highway-mpg'].astype(float)
st.dataframe(df[['city-mpg', 'city-L/100km', 'highway-mpg', 'highway-L/100km']].head())

# Data Normalization
st.subheader("Normalize Dimensions")
df['length'] = df['length'].astype(float) / df['length'].astype(float).max()
df['width'] = df['width'].astype(float) / df['width'].astype(float).max()
df['height'] = df['height'].astype(float) / df['height'].astype(float).max()

# Binning Horsepower
st.subheader("Horsepower Binning")
df['horsepower'] = df['horsepower'].astype(int)
bins = np.linspace(min(df['horsepower']), max(df['horsepower']), 4)
group_names = ['Low', 'Medium', 'High']
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True)

st.bar_chart(df['horsepower-binned'].value_counts())

# Indicator Variables
st.subheader("Indicator Variables")
df = pd.concat([df, pd.get_dummies(df['fuel-type'], prefix='fuel-type')], axis=1)
df.drop("fuel-type", axis=1, inplace=True)

aspiration_dummies = pd.get_dummies(df['aspiration'], prefix='aspiration')
df = pd.concat([df, aspiration_dummies], axis=1)
df.drop("aspiration", axis=1, inplace=True)

# Final Shape
st.subheader("Final Data Shape")
st.write("Rows and Columns:", df.shape)

# Export Option
st.download_button("Download Cleaned CSV", data=df.to_csv(index=False), file_name='clean_df.csv', mime='text/csv')
