# automobile_app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="Automobile Data Analysis", layout="wide")
st.title("Automobile Data Analysis - Part 1 & Part 2")
st.markdown("**Name**: Rejeti Spandana Sharon  \n**Banner ID**: 001409139")

# Load Data
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/klamsal/Fall2024Exam/refs/heads/main/CleanedAutomobile.csv'
    return pd.read_csv(url)

df = load_data()

# ========================== PART 1 ==========================
st.header("📊 Part 1: Data Preprocessing and Exploration")

st.subheader("1. Raw Data Preview")
st.dataframe(df.head())

st.subheader("2. Data Cleaning Notes")
st.markdown("""
- Missing values were handled.
- Data types were adjusted for numerical analysis.
- Relevant features were retained.
""")

st.subheader("3. Summary Statistics")
st.write("Numerical:")
st.dataframe(df.describe())
st.write("Categorical:")
st.dataframe(df.describe(include="object"))

st.subheader("4. Unique Values in Columns")
selected_col = st.selectbox("Choose a column to see unique values", df.columns)
st.write(df[selected_col].unique())

# ========================== PART 2 ==========================
st.header("📈 Part 2: Statistical and Visual Analysis")

st.subheader("1. Data Types")
st.dataframe(df.dtypes)

st.info(f"**Data type of 'peak-rpm'**: `{df['peak-rpm'].dtype}`")

st.subheader("2. Correlation Matrix for Select Features")
cols = ['bore', 'stroke', 'compression-ratio', 'horsepower']
corr_matrix = df[cols].astype(float).corr()
st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'))

st.subheader("3. Scatter Plot with Regression Line")
feature = st.selectbox("Choose feature to plot against price", df.select_dtypes(include='number').columns)
fig1, ax1 = plt.subplots()
sns.regplot(x=feature, y='price', data=df, ax=ax1)
st.pyplot(fig1)

if feature != "price":
    corr_val = df[[feature, "price"]].corr().iloc[0,1]
    st.write(f"**Correlation between `{feature}` and `price`**: {corr_val:.3f}")

st.subheader("4. Boxplot: Categorical Feature vs Price")
cat_feature = st.selectbox("Select a categorical feature", df.select_dtypes(include='object').columns)
fig2, ax2 = plt.subplots()
sns.boxplot(x=cat_feature, y="price", data=df, ax=ax2)
plt.xticks(rotation=45)
st.pyplot(fig2)

st.subheader("5. Grouping and Pivot Table Analysis")
avg_price_by_body = df.groupby("body-style")["price"].mean().reset_index()
st.write("Average Price by Body Style")
st.dataframe(avg_price_by_body)

grouped = df[['drive-wheels', 'body-style', 'price']].groupby(['drive-wheels', 'body-style'], as_index=False).mean()
pivot = grouped.pivot(index='drive-wheels', columns='body-style', values='price').fillna(0)

fig3, ax3 = plt.subplots()
im = ax3.imshow(pivot.values, cmap='coolwarm')
ax3.set_xticks(np.arange(len(pivot.columns)))
ax3.set_yticks(np.arange(len(pivot.index)))
ax3.set_xticklabels(pivot.columns)
ax3.set_yticklabels(pivot.index)
plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        ax3.text(j, i, f"{pivot.iloc[i, j]:.0f}", ha="center", va="center", color="black")
fig3.colorbar(im)
st.pyplot(fig3)

st.subheader("6. Pearson Correlation & P-Value")
x = st.selectbox("X Variable", df.select_dtypes(include='number').columns, key='pearson_x')
y = st.selectbox("Y Variable", df.select_dtypes(include='number').columns, key='pearson_y')

if x and y:
    pearson_coef, p_value = stats.pearsonr(df[x], df[y])
    st.write(f"**Pearson Correlation Coefficient** between `{x}` and `{y}`: {pearson_coef:.3f}")
    st.write(f"**P-value**: {p_value:.5f}")
    if p_value < 0.001:
        st.success("Strong evidence the correlation is significant.")
    elif p_value < 0.05:
        st.info("Moderate evidence the correlation is significant.")
    elif p_value < 0.1:
        st.warning("Weak evidence the correlation is significant.")
    else:
        st.error("No evidence that the correlation is significant.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
