import streamlit as st
import requests
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

st.title('Monitoring Dashboard')
image = Image.open('./assets/stats.png')
st.image(image)
st.markdown("###### A simple tool for monitoring the performance of our model. This simple monitoring dashboard will help us track the inference latency and evaluate trends in prediction results.")

st.markdown("### Record of Inference Results")
st.caption("A table containing metadata about each inference request made.")

# Logic for inference metadata table

st.divider()

st.markdown("### Chart of Inference Time in Milliseconds (ms) vs Request DateTime Stamps")
st.caption("A line graph depicting the change inference time over time. ")

# Logic for inference latency line chart

st.divider()

st.markdown("### Chart of Predicted Labels vs Request DateTime Stamps")
st.caption("A plot depicting the change predictions over time. ")

# Logic for predictions over time

st.divider()

st.markdown("### Histogram of Results")
st.caption("A histogram showing the frequency of each prediction label.")

# Logic for predictions histogram