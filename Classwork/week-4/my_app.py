import streamlit as st

st.title("Hello, streamlit!")

st.write("This is my first Streamlit app.")

if st.button("Click me!"):
    st.write("You clicked the button!")
else: 
    st.write("Click the button and see what happens...")

import pandas as pd

st.subheader("Exploring Our Dataset")

df = pd.read_csv("data/sample_data.csv")

st.write("Here's our data!")
st.dataframe(df)

city = st.selectbox("Select a city", df["City"].unique(), index = None)
filtered_df = df[df["City"] == city]

st.write(f"People in {city}")
st.dataframe(filtered_df)

st.bar_chart(df["Salary"])

import seaborn as sns

box_plot1 = sns.boxplot(x = df['City'], y = df['Salary'])

st.pyplot(box_plot1.get_figure())