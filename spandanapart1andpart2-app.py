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
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# ✅ Must be first Streamlit command
st.set_page_config(layout="wide")

# Title and Info
st.title("Exam Part 2 - Automobile Price Analysis")
st.markdown("""
**Name:** Rejeti Spandana Sharon  
**Banner ID:** 001409139
""")

# Load Dataset
st.header("1. Import Data")
url = 'https://raw.githubusercontent.com/klamsal/Fall2024Exam/main/CleanedAutomobile.csv'
df = pd.read_csv(url)
st.dataframe(df.head())

# Data Types
st.subheader("Data Types")
st.write(df.dtypes)

# Question 1: Data Type of 'peak-rpm'
st.info(f"Data type of 'peak-rpm': {df['peak-rpm'].dtype}")

# Question 2: Correlation
st.subheader("Correlation: bore, stroke, compression-ratio, horsepower")
df_corr = df[['bore', 'stroke', 'compression-ratio', 'horsepower']].astype(float)
st.dataframe(df_corr.corr())

# Scatterplots and Correlations
st.header("2. Pattern Visualization")

with st.expander("🔍 Scatterplots and Correlations"):
    fig1 = sns.regplot(x="engine-size", y="price", data=df)
    st.pyplot(fig1.figure)
    st.write("Correlation (engine-size vs price):", df[['engine-size', 'price']].corr().iloc[0,1])

    fig2 = sns.regplot(x="highway-mpg", y="price", data=df)
    st.pyplot(fig2.figure)
    st.write("Correlation (highway-mpg vs price):", df[['highway-mpg', 'price']].corr().iloc[0,1])

    fig3 = sns.regplot(x="peak-rpm", y="price", data=df)
    st.pyplot(fig3.figure)
    st.write("Correlation (peak-rpm vs price):", df[['peak-rpm', 'price']].corr().iloc[0,1])

# Question 3a and 3b
st.subheader("Q3: Stroke vs Price")
st.write("Correlation:")
st.write(df[["stroke", "price"]].astype(float).corr())

fig4 = sns.regplot(x="stroke", y="price", data=df)
st.pyplot(fig4.figure)

# Categorical Boxplots
st.header("3. Categorical Variable Analysis")
st.subheader("Boxplot: Body Style vs Price")
fig5 = sns.boxplot(x="body-style", y="price", data=df)
st.pyplot(fig5.figure)

st.subheader("Boxplot: Engine Location vs Price")
fig6 = sns.boxplot(x="engine-location", y="price", data=df)
st.pyplot(fig6.figure)

st.subheader("Boxplot: Drive Wheels vs Price")
fig7 = sns.boxplot(x="drive-wheels", y="price", data=df)
st.pyplot(fig7.figure)

# Descriptive Stats
st.header("4. Descriptive Statistics")
st.subheader("Numerical Description")
st.dataframe(df.describe())

st.subheader("Categorical Description")
st.dataframe(df.describe(include=['object']))

# Value Counts
st.subheader("Drive Wheels Value Counts")
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts.index.name = 'drive-wheels'
st.dataframe(drive_wheels_counts)

# Grouping
st.header("5. Grouping Data")
df_group = df[['drive-wheels', 'price']]
df_grouped = df_group.groupby(['drive-wheels'], as_index=False).mean()
st.subheader("Average Price by Drive Wheels")
st.dataframe(df_grouped)

# Question 4: Group by body-style
st.subheader("Q4: Average Price by Body Style")
avg_price_by_body = df.groupby("body-style")["price"].mean().reset_index()
st.dataframe(avg_price_by_body)

# Heatmap
grouped_test1 = df[['drive-wheels','body-style','price']].groupby(['drive-wheels','body-style'],as_index=False).mean()
grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style', values='price').fillna(0)

st.subheader("Heatmap: Drive Wheels and Body Style vs Price")
fig, ax = plt.subplots(figsize=(10,6))
im = ax.pcolor(grouped_pivot, cmap='RdBu')
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5)
ax.set_xticklabels(grouped_pivot.columns, rotation=90)
ax.set_yticklabels(grouped_pivot.index)
fig.colorbar(im)
st.pyplot(fig)

# Correlation & Causation
st.header("6. Correlation and Causation")
def corr_pval(x, y):
    coef, p = stats.pearsonr(df[x], df[y])
    return coef, p

columns_to_check = ['wheel-base', 'horsepower', 'length', 'width', 'curb-weight',
                    'engine-size', 'bore', 'city-mpg', 'highway-mpg']

for col in columns_to_check:
    coef, p = corr_pval(col, 'price')
    st.markdown(f"**{col} vs price** → Pearson Correlation: {coef:.3f}, P-value: {p:.5f}")
