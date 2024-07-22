from mobileprice.entity import DataTransformationConfig
import os
import pandas as pd
import numpy as np
from mobileprice import logger
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
    
    def FeatureSelection(self,X,y):
        '''Feature Selection'''
        logger.info('Feature Selection Using Lasso Regression Model ')
        feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0)) # remember to set the seed, the random state in this function
        feature_sel_model.fit(X, y)
        selected_feat = X.columns[(feature_sel_model.get_support())]

        # let's print some stats
        logger.info('Total Features: {}'.format((X.shape[1])))
        logger.info('Selected Features: {}'.format(len(selected_feat)))
        logger.info('Features With Co-Efficients Shrank to Zero: {}'.format(
            np.sum(feature_sel_model.estimator_.coef_ == 0)))
        logger.info("Completed Feature Selection")
        return selected_feat

    def scaling(self,dataset):
        '''Scaling Feature'''
        
        logger.info('Scaling down Selected Features')
        scaling_feature=[feature for feature in dataset.columns if feature not in [self.config.target_column] ]
        scaler=MinMaxScaler()
        scaler.fit(dataset[scaling_feature])
        data = pd.concat([dataset[[self.config.target_column]].reset_index(drop=True),
                    pd.DataFrame(scaler.transform(dataset[scaling_feature]), columns=scaling_feature)],
                    axis=1)
        logger.info("Completed Scaling Dataset")
        return(data)
        
    def train_test_spliting(self,data):
        '''  Train-Test split into csv file'''

        logger.info('Train-Test split into CSV File')
        
        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info("train: {}".format(train.shape))
        logger.info("test: {}".format(test.shape))

        print(train.shape)
        print(test.shape)
        
    
    def transformation(self):
        data = pd.read_csv(self.config.data_path)
        logger.info("Converted CSV data to DataFrame")
    
        #Feature Selection
        selected_feat=self.FeatureSelection(data.drop([self.config.target_column],axis=1),
                                            data[[self.config.target_column]]
                                            )
        data=data[selected_feat].join(data[self.config.target_column],how='inner')
    
        
        
        #scaling the dependent variable
        data=self.scaling(data)


        #Train test split into csv file
        self.train_test_spliting(data)
