from distutils.errors import PreprocessError
import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transfromation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        '''
            This fucntion is responsible for data tranformation
        '''
        try:
            numercal_columns = ['writing_score' , 'reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]
            num_pipeline = Pipeline(
                steps = [
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

            ]
            )
            logging.info('Numerical cols scaling complete')


            cat_pipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))

                ]
            )

            logging.info('Categorical cols encoding complete')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numercal_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns),
                ]
            )
            logging.info('preprocessor pipeline creation done')

            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train/test')

            logging.info('Obtainng preprocessing object')

            preprocessor_obj  = self.get_data_transformer_obj()

            target_column_name = 'math_score'

            target_feature_train_df = train_df[target_column_name]
            target_feature_test_df = test_df[target_column_name]

            input_feature_train_df = train_df.drop(target_column_name,axis = 1)
            input_feature_test_df = test_df.drop(target_column_name,axis = 1)

            logging.info('Applying Preprocessing to train and test data')

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info('Saving preprocessed obj')

            save_obj(
                file_path = self.data_transfromation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transfromation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)
