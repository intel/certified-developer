import streamlit as st
from PIL import Image


st.title('The Prototype')
st.header('Pharmaceutical Manufacturing Business')
st.markdown('Building a Prototype for the MLOps Certifcation Capstone Project.')

st.divider()

col1, col2 = st.columns(2)

with col1:
   st.subheader("Robotics Maintenance")
   forecasting_image = Image.open('./assets/robot_arm.png')
   st.image(forecasting_image)
   st.caption('Computer vision quality inspection tool to flag and remove bad pills from production line')
   
with col2:
   st.subheader('Monitoring Dashboard')
   forecasting_image = Image.open('./assets/stats.png')
   st.image(forecasting_image)
   st.caption('Customer support chatbot based on pre-trained gpt-j large language model')

st.divider()
   
st.markdown('##### Notices & Disclaimers')
st.caption('Performance varies by use, configuration, and other factors. Learn more on the Performance \
    Index site. Performance results are based on testing as of dates shown in configurations and may not\
        reflect all publicly available updates. See backup for configuration details. No product or component\
            can be absolutely secure. Your costs and results may vary. Intel technologies may require enabled\
                hardware, software, or service activation. © Intel Corporation. Intel, the Intel logo, and other\
                    Intel marks are trademarks of Intel Corporation or its subsidiaries. Other names and brands may\
                        be claimed as the property of others.')
   