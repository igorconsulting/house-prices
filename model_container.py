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

class ModelContainer:
    
    def __init__(self):
        '''initializes model list and dicts'''
        self.best_algorithm  = None
        self.gridcvs = {}
        self.scores_dict = None
        self.best_model = None
        self.best_params  = None
        self.best_score  = 0
        self.predictions = None
        self.train_mae = 0
        self.test_mae  = 0
        self.train_r2  = 0
        self.test_r2   = 0
        self.mean_rmse = {}
        self.parameters = {}
    
        
    def nested_cross_validation(self, features, target):
        '''This function performs the nested 5x2cv procedure and selects best algorithm'''    
        reg_RF = RandomForestRegressor(random_state=1)
        reg_XGB = xgb.XGBRegressor(random_state=1)
        reg_LGBM = LGBMRegressor(random_state=1)   
                   
        param_grid_RF = {
                        'bootstrap': [True],
                        'max_depth': [80, 90, 100, 110],
                        'max_features': [2, 3],
                        'min_samples_leaf': [3, 4, 5],
                        'min_samples_split': [8, 10, 12],
                        'n_estimators': [100, 200, 300, 1000]
                        }
        
        param_grid_XGB = {
                        'min_child_weight': [1, 5, 10],
                        'gamma': [0.5, 1, 1.5, 2, 5],
                        'subsample': [0.6, 0.8, 1.0],
                        'colsample_bytree': [0.6, 0.8, 1.0],
                        'max_depth': [3, 4, 5]
                        }
        
        param_grid_LGBM = {'num_leaves': [6, 8, 20, 30],
                        'max_depth': [2, 4, 6, 8, 10],
                        'n_estimators': [50, 100, 200, 500],
                        'colsample_bytree': [0.3, 1.0]}

        
        self.parameters= {reg_RF:param_grid_RF, reg_XGB:param_grid_XGB, reg_LGBM:param_grid_LGBM}
        
    
        inner_cv = KFold(n_splits=2, shuffle=True, random_state=1)

        for pgrid, est, name in zip((param_grid_RF, param_grid_XGB, param_grid_LGBM),
                                    (reg_RF, reg_XGB, reg_LGBM),
                                    ('RForest','Xgboost', 'LightGBM')):

            gcv = GridSearchCV(estimator=est,
                               param_grid=pgrid,
                               scoring = 'neg_root_mean_squared_error',
                               n_jobs=-1,
                               cv=inner_cv,
                               verbose=0,
                               refit=True)
            self.gridcvs[name] = gcv


        outer_cv = KFold(n_splits=5, shuffle=True, random_state=1)
    
        for name, gs_est in sorted(self.gridcvs.items()):
            self.scores_dict = cross_validate(gs_est, 
                                         X=features, 
                                         y=target,
                                         verbose=0,
                                         cv=outer_cv,
                                         return_estimator=True,
                                         n_jobs=-1
                                        )

            print(50 * '-', '\n')
            print('Algorithm:', name)
            print('    Inner loop:')


            for i in range(self.scores_dict['test_score'].shape[0]):
                print('\n      Best RMSE Score (avg. of inner test folds) %.2f' % np.absolute(self.scores_dict['estimator'][i].best_score_))
                print('        Best parameters:', self.scores_dict['estimator'][i].best_estimator_)
                print('        RMSE Score (on outer test fold) %.2f' % np.absolute(self.scores_dict['test_score'][i]))
            print('\n%s |  outer test folds Ave. Score %.2f +/- %.2f' % 
                                  (name, np.absolute(self.scores_dict['test_score']).mean(), 
                                   np.absolute(self.scores_dict['test_score']).std()))    
            
            self.mean_rmse[gs_est] = np.absolute(self.scores_dict['test_score']).mean() 
            
        self.best_algorithm = min(self.mean_rmse, key=self.mean_rmse.get)
        print ('\nBest Performing Algorithm: ', self.best_algorithm.estimator)
    

   
    def tune_best_algorithm (self, feature_train, feature_test, target_train, target_test): 
        '''This function performs hyperparameter tuning on the whole training set with the best algorithm '''
        gcv_model_select = GridSearchCV(estimator=self.best_algorithm.estimator,
                                        param_grid=self.parameters[self.best_algorithm.estimator],
                                        scoring='neg_root_mean_squared_error',
                                        n_jobs=-1,
                                        cv = 2,
                                        verbose=0,
                                        refit=True)

        gcv_model_select.fit(feature_train, target_train)
            
        self.best_model = gcv_model_select.best_estimator_
        self.best_score = gcv_model_select.best_score_
        self.best_params = gcv_model_select.best_params_
            
        self.train_mae = mean_absolute_error(y_true=np.exp(target_train), y_pred=np.exp(self.best_model.predict(feature_train)))
        self.test_mae  = mean_absolute_error(y_true=np.exp(target_test),  y_pred=np.exp(self.best_model.predict(feature_test)))

        self.train_r2 = r2_score (y_true=np.exp(target_train), y_pred=np.exp(self.best_model.predict(feature_train)))
        self.test_r2  = r2_score (y_true=np.exp(target_test),  y_pred=np.exp(self.best_model.predict(feature_test)))

   

    def best_model_predict(self, features):
        '''scores features using best model'''
        self.predictions = self.best_model.predict(features)
       
    
    def save_results(self):
        pass 
    
        
    @staticmethod
    def get_feature_importance(model, cols):
        '''retrieves and sorts feature importances'''
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            cols = model.feature_name_
            feature_importances = pd.DataFrame({'feature':cols, 'importance':importances})
            feature_importances.sort_values(by='importance', ascending=False, inplace=True)
            #set index to 'feature'
            feature_importances.set_index('feature', inplace=True, drop=True)
            return feature_importances
        else:
            #some models don't have feature_importances_
            return "Feature importances do not exist for given model"

        
    def print_summary(self):
        '''prints summary of models, best model, and feature importance'''
        print('\nModel Summaries:\n')
        
        print('Best Estimator:' ,self.best_model)
        print('Best CV Score: %.2f' % np.abs(self.best_score))
        print('Best Parameters: %s' % self.best_params)
        
        print('\nTrain MAE: %.2f' % self.train_mae)
        print(' Test MAE: %.2f' %  self.test_mae)

        print('\nTrain R2: %.2f' % self.train_r2)
        print(' Test R2: %.2f' %   self.test_r2)
            
        feature_importances = self.get_feature_importance(self.best_model, data.feature_cols)
        feature_importances[0:25].plot.bar(figsize=(20,10))
        plt.show()
    
    
    
    def save_best_model(self, filepath = 'best_model.pkl'):
        """
        Save a regression model to a file.

        Parameters:
        - model: The regression model object.
        - filepath (str): The path to save the model file.

        Returns:
        - None

        Example Usage:
        >>> model = YourRegressionModel()
        >>> # Train the model...
        >>> save_regression_model(model, 'model.pkl')
        """
        joblib.dump(self.best_model, filepath)