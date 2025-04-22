import streamlit as st
from sklearn.datasets import load_iris

import pandas as pd
import numpy as np

st.write("Hello World")
x = st.text_input("Favourite Movie?")
st.write(f"Your favourite movie is {x}")

is_clicked = st.button("Click me")

if is_clicked:
    st.write("Cool, right?")

st.write("## This is a H2 Title!")

# Load iris dataset
#iris = load_iris()

#st.write(iris['data'])

chart_data = pd.DataFrame(
    np.random.randn(20,3)

    ,columns=['a','b','c']
)

st.bar_chart(chart_data)
st.line_chart(chart_data)