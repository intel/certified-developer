## Code Snippets for Completing Capstone

```python
# Part 1 - Connecting Training FE/BE Target File: RoboMaintenance.py
if st.button('Train Model', key='training'):
        # build request

        URL = 'http://localhost:5000/train'
        DATA = {'file':data_file, 'model_name':model_name, 'model_path':model_path, 
                  'test_size': test_size, 'ncpu': 4, 'mlflow_tracking_uri':mlflow_tracking_uri,
                  'mlflow_new_experiment':mlflow_new_experiment, 'mlflow_experiment':mlflow_experiment}
        TRAINING_RESPONSE = requests.post(url = URL, json = DATA)

        if len(TRAINING_RESPONSE.text) < 40:       
            st.error("Model Training Failed")
            st.info(TRAINING_RESPONSE.text)
        else:
            st.success('Training was Succesful')
            st.info('Model Validation Accuracy Score: ' + str(TRAINING_RESPONSE.json().get('validation scores')))

# Part 2 - Connecting Inference FE/BE Target File: RoboMaintenance.py
if st.button('Run Maintenance Analysis', key='analysis'):
        URL = 'http://localhost:5000/predict'
        DATA = {'model_name':model_name, 'stage':stage, 'sample':sample, 
                'model_run_id':model_run_id, 'scaler_file_name':scaler_file_name, 'scaler_destination':scaler_destination}
        INFERENCE_RESPONSE = requests.post(url = URL, json = DATA)

        if len(INFERENCE_RESPONSE.text) < 40:       
            st.error("Model Inference Failed")
            st.info(INFERENCE_RESPONSE.text)
        else:
            st.success(str(INFERENCE_RESPONSE.json().get('Maintenance Recommendation')))

# Part 3 - Add Monitoring Log File Logic Target File: inference.py
current_datetime = datetime.datetime.now()
current_datetime_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
data_dict = {'model':model_name, 'stage': stage, 
             'model_run_id': model_run_id, 'scaler_file_name': scaler_file_name, 
             'prediction': prediction, 'inference_time': elapsed_time_milliseconds, 'datetime': current_datetime_str}
file_path = scaler_destination + '/monitoring.csv'


if os.path.isfile(file_path):
    df = pd.read_csv(file_path)
else:
    df = pd.DataFrame(columns=data_dict.keys())

df = pd.concat([df, pd.DataFrame(data_dict, index=[0])], ignore_index=True)
df.to_csv(file_path, index=False)

# Part 4 - Add Monitoring Logic Target File: Monitoring.py
df = pd.read_csv(r'/home/ubuntu/certified-developer/MLOps_Professional/mlops_capstone/store/outputs/robot_maintenance/monitoring.csv')
df

st.line_chart(data=df, x='datetime', y='inference_time')

st.scatter_chart(data=df, x='datetime', y='prediction')

fig, ax = plt.subplots()
ax.hist(df['prediction'])
st.pyplot(fig)
```
