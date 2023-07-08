# scientifit libraries
import pandas as pd
import pandas as pd
import math 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import warnings
warnings.filterwarnings('ignore')

# sklearn transformers
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder

from feature_engine import encoding as enc #RareLabelEncoder, OrdinalEncoder
from feature_engine.selection import SelectByTargetMeanPerformance, DropFeatures 
from feature_engine.creation import RelativeFeatures, MathFeatures
from feature_engine.creation import RelativeFeatures, MathFeatures
# sk-learn pipeline
from sklearn.pipeline import Pipeline

# sampling
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

# hyperparameter tuning
from sklearn.model_selection import GridSearchCV

# metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score

# Regression models
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor


# graphical libraries
import seaborn as sns
import matplotlib.pyplot as plt

# a library to analyze missing values
import missingno as msno

class DataPreprocessing:
    """
    Preprocess train and test dataframes.

    Attributes:
    - train_data (pandas.DataFrame): The dataframe for the train data.
    - test_data (pandas.DataFrame): The dataframe for the test data.
    - nominal_columns (list): The list of nominal columns.
    - ordinal_columns (list): The list of ordinal columns.
    - ordinal_categories_list (list): The list of ordinal categories for ordinal encoding.
    - numerical_columns (list): The list of numerical columns.
    - target_column (str): The name of the target column.
    - categorical_columns (list): The list of categorical columns (nominal and ordinal).
    - feature_columns (list): The list of all feature columns (categorical and numerical).
    - cols_to_drop (list): The list of columns to drop.
    - ordinal_encoders (dict): The dictionary of ordinal encoders.
    - encoder: The encoder for ordinal encoding.

    Methods:
    - _load_data(file): Loads a CSV file into a pandas DataFrame.
    - _log_transform(df): Performs a log transformation of the target column.
    - _drop_missing_cols(df): Identifies and drops columns with a high proportion of missing data.
    - _impute_missing_values(df, categorical_features, numeric_features): Imputes missing values using strategies.
    - _ordinal_encode(df, ordinal_columns, ordinal_categories_list): Encodes ordinal variables using ordinal encoding.
    - _inverse_ordinal_encode(df, ordinal_columns): Inverse transforms ordinal encoded columns.
    - _create_train_data(train_file, preprocess=True): Loads and encodes the train data.
    - _create_test_data(test_file, preprocess=True): Loads and encodes the test data.
    - _convert_month_string(df): Maps numerical month names to string month names.
    - _convert_data_types(df): Converts categorical and numerical columns to the appropriate data types.
    """


    def __init__(self, train_file,
                  test_file,
                  nominal_columns,
                  ordinal_columns,
                  ordinal_categories_list,
                  numerical_columns,
                  target_column):
        #create new copies instead of references
        self.nominal_columns = list(nominal_columns)
        self.ordinal_columns = list(ordinal_columns)
        self.numerical_columns = list(numerical_columns)
        self.target_col = target_column
        self.ordinal_categories_list = list(ordinal_categories_list)
        self.categorical_columns = self.nominal_columns + self.ordinal_columns
        self.feature_cols = self.categorical_columns + self.numerical_columns + self.ordinal_columns
        self.cols_to_drop =[]
        self.ordinal_encoders = {}
        self.train_data = self._create_train_data(train_file)
        self.test_data  = self._create_test_data(test_file)
        self.encoder = None

    def _load_data(self,file):
            '''loads csv to pd dataframe'''
            return pd.read_csv(file)


    def _log_transform (self, df):
        '''This function performs the log transformation of the target'''
        df['SalePrice'] = np.log(df['SalePrice'])
        return df
    def _drop_missing_cols (self,df):
        '''Identifies and drops the columns with 80% or hihgher proportion of missing data '''
        dropped_cols = []  
        for col in df.columns:
            if df[col].isnull().sum()/df.shape[0] >= 0.8:
                dropped_cols.append(col)
        df.drop(columns=dropped_cols, inplace=True)
        return df, dropped_cols 
    def _impute_missing_values (self, df, categorical_features, numeric_features):
        ''' Imputes the continious columns with median and categorical columns with the mode value'''
        imputer_con = SimpleImputer(missing_values=np.nan, strategy='median')
        imputer_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        for col in categorical_features+numeric_features:
            if df[col].isnull().sum() > 0:    
                if col in categorical_features:              
                    df[col] = imputer_cat.fit_transform(df[col].values.reshape(-1,1))
                elif col in numeric_features:  
                    df[col] = imputer_con.fit_transform(df[col].values.reshape(-1,1))
        return df 


    def _ordinal_encode (self, df, ordinal_columns, ordinal_categories_list):
        '''This function encodes ordinal variables into ordinal encoding and combines wit the rest of the dataframe'''
        encoder = OrdinalEncoder(categories=ordinal_categories_list)
        df[ordinal_columns] = encoder.fit_transform(df[ordinal_columns])
        return df


    def _inverse_ordinal_encode (self, df, ordinal_columns):
         df[ordinal_columns] = self.encoder.inverse_transform(df[ordinal_columns]) 


    def _create_train_data(self, train_file, preprocess=True):
        '''loads and encodes train data'''
        train_data = self._load_data(train_file)
        if preprocess:
            train_data = self._log_transform(train_data)
            train_data, self.cols_to_drop = self._drop_missing_cols(train_data)
            train_data = self._impute_missing_values(train_data, self.categorical_columns, self.numerical_columns)
            train_data = self._convert_month_string(train_data)
            train_data = self._ordinal_encode(train_data, self.ordinal_columns, self.ordinal_categories_list)
            train_data = self._convert_data_types(train_data)
        return train_data

    def _create_test_data (self,test_file, preprocess=True):
        '''loads and ordinal encodes test data'''
        test_data = self._load_data(test_file)
        if preprocess:
            test_data = test_data.drop(columns=self.cols_to_drop, axis=1)
            test_data = self._impute_missing_values(test_data, self.categorical_columns, self.numerical_columns)
            test_data = self._convert_month_string(test_data)
            test_data = self._ordinal_encode(test_data, self.ordinal_columns, self.ordinal_categories_list) 
            test_data = self._convert_data_types(test_data)
        return test_data

    def _convert_month_string (self, df):
        '''This function maps the numerical month names into string month names'''
        d = { 1 : 'Jan',
              2 : 'Feb',
              3 : 'Mar',
              4 : 'Apr',
              5 : 'May',
              6 : 'June',
              7 : 'July',
              8 : 'Aug',
              9 : 'Sep',
              10: 'Oct',
              11: 'Nov',
              12: 'Dec'
        }
        df['MoSold'] = df ['MoSold'].map(d)
        return df
    def _convert_data_types (self, df):
        '''This function coverts the categorical variables into object and numeric variables into int types'''
        df[self.nominal_columns] = df[self.nominal_columns].astype('O')
        df[self.ordinal_columns] = df[self.ordinal_columns].astype('int')
        df[self.numerical_columns] = df[self.numerical_columns].astype('int') 
        return df