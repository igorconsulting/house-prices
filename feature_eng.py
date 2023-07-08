import warnings
warnings.filterwarnings('ignore')

from feature_engine import encoding as enc #RareLabelEncoder, OrdinalEncoder
from feature_engine.selection import SelectByTargetMeanPerformance, DropFeatures 
from feature_engine.creation import RelativeFeatures, MathFeatures
from feature_engine.creation import RelativeFeatures, MathFeatures

# sk-learn pipeline
from sklearn.pipeline import Pipeline

# metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score


class FeatureEngineering:
    
    def __init__(self, data):
        '''creates feature generator pipeline'''
        self.data = data
        self.categorical_columns = data.categorical_columns
        self.pipeline = None
    
    def create_pipeline (self, features, target):
        ''' Creates engineer various features using pipeline'''
        rare_encoder = enc.RareLabelEncoder(tol = 0.05, n_categories=4, variables = data.nominal_columns)
        price_encoder = enc.OrdinalEncoder (encoding_method='ordered',  variables = data.nominal_columns)

        age = RelativeFeatures(
            variables=['YrSold'],
            reference=['YearBuilt', 'YearRemodAdd', 'GarageYrBlt'],
            func = ['sub']
        )     

        bath = MathFeatures(
            variables=['BsmtHalfBath', 'BsmtFullBath', 'FullBath', 'HalfBath'],
            func=['sum'],
            new_variables_names=['TotalBath'],
        )

        area = MathFeatures(
            variables=['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF'],
            func=['sum'],
            new_variables_names=['TotalArea'],
        )

        drop = DropFeatures(
            features_to_drop=['YearBuilt','YrSold','YearRemodAdd', 'GarageYrBlt']
        )

        pipe = Pipeline(steps=[ 
                                ('rare_encoder', rare_encoder), 
                                ('ordinal_encoder', price_encoder),
                                ('cobinator',age),
                                ('bath', bath),
                                ('area', area),
                                ('drop', drop)
                            ])
        self.pipeline = pipe.fit(features, target)  
    
    def create_features(self,df):        
        '''Transform data and Generate Features'''
        df = self.pipeline.transform (df)
        return df