import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Set page config
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
