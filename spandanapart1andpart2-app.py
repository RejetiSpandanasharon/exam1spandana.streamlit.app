import streamlit as st
# ✅ Must be first Streamlit command
st.set_page_config(layout="wide")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Title and Info
st.title("Automobile Data Analysis - Exam Part 1 & 2")
st.markdown("""
**Name:** Rejeti Spandana Sharon  
**Banner ID:** 001409139
""")

# Load Dataset
st.header("📥 1. Import Cleaned Automobile Data")
url = 'https://raw.githubusercontent.com/klamsal/Fall2024Exam/main/CleanedAutomobile.csv'
df = pd.read_csv(url)
st.dataframe(df.head())

# PART 1 - Data Preprocessing Summary
st.header("🧹 Part 1: Data Cleaning, Normalization, and Encoding Summary")
st.markdown("""
- Missing values in numerical columns were replaced with the **mean** (e.g., `normalized-losses`, `bore`, `stroke`, `horsepower`, `peak-rpm`).  
- Missing values in categorical columns like `num-of-doors` were filled using the **most frequent** value.  
- Rows with missing target values (like `price`) were **dropped**.  
- Data types were converted appropriately (e.g., `bore` to `float`, `normalized-losses` to `int`).
- **Normalization** was applied to `length`, `width`, and `height` to scale values between 0 and 1.  
- Fuel consumption was converted from `mpg` to `L/100km`.  
- Indicator variables were created for `fuel-type` and `aspiration`.
""")

# PART 2 Starts Here
# Data Types
st.header("🔍 2. Data Exploration and Visualization")
st.subheader("Data Types")
st.write(df.dtypes)

st.subheader("Q1: Data Type of 'peak-rpm'")
st.info(f"Data type of 'peak-rpm': {df['peak-rpm'].dtype}")

st.subheader("Q2: Correlation Matrix - Selected Features")
df_corr = df[['bore', 'stroke', 'compression-ratio', 'horsepower']].astype(float)
st.dataframe(df_corr.corr())

# Scatterplots and Correlations
st.subheader("Visual Correlation Check")
with st.expander("Scatterplots + Regression Lines"):
    fig1 = sns.regplot(x="engine-size", y="price", data=df)
    st.pyplot(fig1.figure)

    fig2 = sns.regplot(x="highway-mpg", y="price", data=df)
    st.pyplot(fig2.figure)

    fig3 = sns.regplot(x="peak-rpm", y="price", data=df)
    st.pyplot(fig3.figure)

# Question 3a and 3b
st.subheader("Q3: Stroke vs Price")
st.write(df[["stroke", "price"]].astype(float).corr())
fig4 = sns.regplot(x="stroke", y="price", data=df)
st.pyplot(fig4.figure)

# Boxplots
st.header("📦 3. Categorical Variable Analysis")
for cat_var in ["body-style", "engine-location", "drive-wheels"]:
    fig = sns.boxplot(x=cat_var, y="price", data=df)
    st.subheader(f"Boxplot: {cat_var} vs Price")
    st.pyplot(fig.figure)

# Descriptive Stats
st.header("📊 4. Descriptive Statistics")
st.dataframe(df.describe())
st.subheader("Categorical Description")
st.dataframe(df.describe(include=['object']))

# Grouping
st.header("🧮 5. Grouping and Aggregation")
df_grouped = df.groupby(['drive-wheels'], as_index=False)['price'].mean()
st.subheader("Average Price by Drive Wheels")
st.dataframe(df_grouped)

st.subheader("Q4: Average Price by Body Style")
st.dataframe(df.groupby("body-style")["price"].mean().reset_index())

# Heatmap
grouped = df[['drive-wheels','body-style','price']].groupby(['drive-wheels','body-style'],as_index=False).mean()
pivoted = grouped.pivot(index='drive-wheels',columns='body-style', values='price').fillna(0)
fig, ax = plt.subplots(figsize=(10,6))
im = ax.pcolor(pivoted, cmap='RdBu')
ax.set_xticks(np.arange(pivoted.shape[1]) + 0.5)
ax.set_yticks(np.arange(pivoted.shape[0]) + 0.5)
ax.set_xticklabels(pivoted.columns, rotation=90)
ax.set_yticklabels(pivoted.index)
fig.colorbar(im)
st.subheader("Heatmap: Drive Wheels and Body Style vs Price")
st.pyplot(fig)

# Pearson Correlation
st.header("📈 6. Pearson Correlation with P-values")
def corr_pval(x, y):
    coef, p = stats.pearsonr(df[x], df[y])
    return coef, p

columns = ['wheel-base', 'horsepower', 'length', 'width', 'curb-weight',
           'engine-size', 'bore', 'city-mpg', 'highway-mpg']

for col in columns:
    coef, p = corr_pval(col, 'price')
    st.markdown(f"**{col} vs price** → Pearson r = {coef:.3f}, P = {p:.5f}")
