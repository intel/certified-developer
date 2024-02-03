import streamlit as st
import requests
import os
from PIL import Image

st.title('Robotics Predictive Maintenance')

app_tab, help_tab = st.tabs(["Application", "Help"])

with app_tab:

    col11, col22= st.columns(2)

    with col11:
        image = Image.open('./assets/robot_arm.png')
        st.image(image)
    with col22:
        st.markdown("##### The demand predictive asset maintenance component uses an XGBoost classifier to flag assets that need maintenance. It leverages the Intel® Extension for Scikit-Learn, XGBoost, and daal4py on Intel® 4th Generation Xeon® Scalable processors.")

    st.divider()
    
    st.markdown('#### Predictive Maintenance Model Training')
    
    data_file = st.text_input('Training Data File Path',key='data', value='/home/ubuntu/certified-developer/MLOps_Professional/mlops_capstone/store/datasets/robot_maintenance/train.pkl')
    model_name = st.text_input('Model Name',key='model name', help='The name of the model without extensions', value='model')
    model_path = st.text_input('Model Save Path',key='model path', help='Provide the path without file name', value='./')
    test_size = st.slider('Percentage of data saved for Testing',min_value=5, max_value=50, value=25, step=5)
    ncpu = st.number_input('Threads', min_value=2, max_value=16, step=2)
    mlflow_tracking_uri = st.text_input('Tracking URI',key='uri', value='/home/ubuntu/certified-developer/MLOps_Professional/mlops_capstone/store/models/robot_maintenance')
    mlflow_new_experiment = st.text_input('New Experiment Name',key='new exp')
    mlflow_experiment = st.text_input('Existing Experiment Name',key='existing exp') 
    
    # logic for training API connections
     
    st.divider()
    
    st.markdown('#### Predictive Maintenance Analysis')
    
    model_name = st.text_input('Model Name',key='model name option', value='model')
    stage = manufacturer = st.selectbox('Model Stage', options = ['Staging','Production'])
    model_run_id = st.text_input('Run ID',key='model id')
    scaler_file_name = st.text_input('Scaler File Name',key='scalar file', value='model_scaler.joblib')
    scaler_destination = st.text_input('Scaler Destination',key='scalerdest', value= '/home/ubuntu/certified-developer/MLOps_Professional/mlops_capstone/store/outputs/robot_maintenance') 
    
    col21, col22, col23 = st.columns(3)

    manufacturer_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    model_list = ['Gen1', 'Gen2', 'Gen3', 'Gen4']
    lubrication_type_list = ['LTA', 'LTB', 'LTC']
    product_assignment_list = ['PillA', 'PillB', 'PillC']

    with col21:
        manufacturer = st.selectbox('Manufacturer', manufacturer_list)
        generation = st.selectbox('Generation', model_list)
        age = st.number_input('Robot Age', min_value=0, max_value=25, step=1, value=0)

    with col22:
        temperature = st.number_input('Temperature', min_value=50, max_value=300, step=1)
        motor_current = st.number_input('Motor Current', min_value=0.00, max_value=10.00, step=.05, value=5.00)
        lubrication_type = st.selectbox('Lubrication Type', lubrication_type_list)
    with col23:
        last_maintenance = st.number_input('Last Maintenance', min_value=0, max_value=60, step=1)
        num_repairs = st.number_input('Repair Counts', min_value=0, max_value=50, step=1)
        product_assignment = st.selectbox('Pill Product Assignment', product_assignment_list)
        
        
    sample = [{'Age':age, 'Temperature':temperature, 'Last_Maintenance':last_maintenance, 'Motor_Current':motor_current,
       'Number_Repairs':num_repairs, 'Manufacturer':manufacturer, 
       'Generation':generation,'Lubrication':lubrication_type, 'Product_Assignment':product_assignment}]

# logic for inference API connections
            
# Help tab frontend below
    
with help_tab:
    st.markdown("#### Input Descriptions:")
    st.markdown("- Manufacturer: Provide the name of the manufacturer of the robotic arm")
    st.markdown("- Model: Specify the model or specific type of the robotic arm. ")
    st.markdown("- Lubrication Type: Indicate the type of lubrication used in the robotic arm.")
    st.markdown("- Pill Type: Specify the type or category that the robotic arm is assigned to")
    st.markdown("- Age of the Machine: Enter the age or duration of use of the robotic arm.")
    st.markdown("- Motor Current: Provide the current reading from the motor of the robotic arm. ")
    st.markdown("- Temperature of Sensors: Specify the temperature readings from the sensors installed on the robotic arm.")
    st.markdown("- Number of Historic Repairs: Enter the total number of repairs or maintenance activities performed on the robotic arm in the past. ")
    st.markdown("- Last Maintenance Date: Provide the date of the last maintenance activity performed on the robotic arm.")
    st.markdown("#### Code Samples:")
    
    st.markdown("##### Conversion of XGBoost to Daal4py Model")
    daalxgboost_code = '''xgb_model = xgb.train(self.parameters, xgb_train, num_boost_round=100)
        self.d4p_model = d4p.get_gbt_model_from_xgboost(xgb_model)'''
    st.code(daalxgboost_code, language='python')
    
    st.markdown("##### Inference with Daal4py Model")
    daalxgboost_code = '''
    daal_predict_algo = d4p.gbt_classification_prediction(
            nClasses=num_class,
            resultsToEvaluate="computeClassLabels",
            fptype='float')
            
    daal_prediction = daal_predict_algo.compute(data, daal_model)
    '''
    st.code(daalxgboost_code, language='python')
    
    st.markdown('[Visit GitHub Repository for Source Code](https://github.com/intel/AI-Hackathon)')
