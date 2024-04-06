'''
Module to train and prediction using XGBoost Classifier
'''

import sys
import logging
import warnings
import joblib
import mlflow

import numpy as np
import xgboost as xgb
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# !/usr/bin/env python
# coding: utf-8
# pylint: disable=import-error
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class HarvesterMaintenance():
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.file = ''
        self.y_train = ''
        self.y_test = ''
        self.X_train_scaled_transformed = ''
        self.X_test_scaled_transformed = ''
        self.accuracy_scr = ''
        self.model_path = ''
        self.parameters = ''
        self.robust_scaler = ''
        self.run_id = ''
        self.active_experiment = ''
        self.xgb_model = ''
        
    def mlflow_tracking(self, tracking_uri: str = './mlruns', experiment: str = None, new_experiment: str = None):
        
        # sets tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        # creates new experiment if no experiment is specified
        if experiment == None:
            mlflow.create_experiment(new_experiment)
            self.active_experiment = new_experiment
            mlflow.set_experiment(new_experiment)
        else:
            mlflow.set_experiment(experiment)
            self.active_experiment = experiment
        
    def process_data(self, file: str, test_size: int = .25):
        """processes raw data for training

        Parameters
        ----------
        file : str
            path to raw training data
        test_size : int, optional
            percentage of data reserved for testing, by default .25
        """

        # Generating our data
        logger.info('Reading the dataset from %s...', file)
        try:
            data = pd.read_pickle(file)
        except FileNotFoundError:
            sys.exit('Dataset file not found')


        X = data.drop('Asset_Label', axis=1)
        y = data.Asset_Label

        X_train, X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size)

        df_num_train = X_train.select_dtypes(['float', 'int', 'int32'])
        df_num_test = X_test.select_dtypes(['float', 'int', 'int32'])
        self.robust_scaler = RobustScaler()
        X_train_scaled = self.robust_scaler.fit_transform(df_num_train)
        X_test_scaled = self.robust_scaler.transform(df_num_test)

        # Making them pandas dataframes
        X_train_scaled_transformed = pd.DataFrame(X_train_scaled,
                                                  index=df_num_train.index,
                                                  columns=df_num_train.columns)
        X_test_scaled_transformed = pd.DataFrame(X_test_scaled,
                                                 index=df_num_test.index,
                                                 columns=df_num_test.columns)

        del X_train_scaled_transformed['Number_Repairs']

        del X_test_scaled_transformed['Number_Repairs']

        # Dropping the unscaled numerical columns
        X_train = X_train.drop(['Age', 'Temperature', 'Last_Maintenance', 'Motor_Current'], axis=1)
        X_test = X_test.drop(['Age', 'Temperature', 'Last_Maintenance', 'Motor_Current'], axis=1)
        
        X_train = X_train.astype(int)
        X_test = X_test.astype(int)

        # Creating train and test data with scaled numerical columns
        X_train_scaled_transformed = pd.concat([X_train_scaled_transformed, X_train], axis=1)
        X_test_scaled_transformed = pd.concat([X_test_scaled_transformed, X_test], axis=1)

        self.X_train_scaled_transformed = X_train_scaled_transformed.astype(
                                        {'Motor_Current': 'float64'})
        self.X_test_scaled_transformed = X_test_scaled_transformed.astype(
                                        {'Motor_Current': 'float64'})
        

    def train(self, ncpu: int = 4):
        """trains an XGBoost Classifier and Tracks Models with MLFlow

        Parameters
        ----------
        ncpu : int, optional
            number of CPU threads used for training, by default 4
        """
        
        # Set xgboost parameters
        self.parameters = {
        'max_bin': 256,
        'scale_pos_weight': 2,
        'lambda_l2': 1,
        'alpha': 0.9,
        'max_depth': 8,
        'num_leaves': 2**8,
        'verbosity': 0,
        'objective': 'multi:softmax',
        'learning_rate': 0.3,
        'num_class': 3,
        'nthread': ncpu
        }
        
        with mlflow.start_run() as run:
            mlflow.xgboost.autolog()   
            xgb_train = xgb.DMatrix(self.X_train_scaled_transformed, label=np.array(self.y_train))
                      
        self.xgb_model = xgb.train(self.parameters, xgb_train, num_boost_round=100)
        

    def validate(self):
        """performs model validation with testing data

        Returns
        -------
        float
            accuracy metric
        """
        dtest = xgb.DMatrix(self.X_test_scaled_transformed, self.y_test)
        xgb_prediction = self.xgb_model.predict(dtest)
        xgb_errors_count = np.count_nonzero(xgb_prediction - np.ravel(self.y_test))
        self.accuracy_scr = 1 - xgb_errors_count / xgb_prediction.shape[0]
        
        xp = mlflow.get_experiment_by_name(self.active_experiment)._experiment_id
        self.run_id = mlflow.search_runs(xp, output_format="list")[0].info.run_id
        
        with mlflow.start_run(self.run_id ):
            mlflow.log_metric("accuracy",self.accuracy_scr)
        
        return self.accuracy_scr
    
    def save(self, model_path):
        """saves trained model to path

        Parameters
        ----------
        model_path : str
            path where trained model should be saved
        """

        self.model_path = model_path +  self.model_name + '.joblib'
        self.scaler_path = model_path +  self.model_name + '_scaler.joblib'
        
        logger.info("Saving model")
        with open( self.model_path, "wb") as fh:
            joblib.dump(self.xgb_model, fh.name)
        
        logger.info("Saving Scaler")
        with open( self.scaler_path, "wb") as fh:
            joblib.dump(self.robust_scaler, fh.name)
