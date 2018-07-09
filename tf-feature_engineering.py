# coding=utf-8
# https://github.com/random-forests/tensorflow-workshop/blob/master/archive/examples/07_structured_data.ipynb

import numpy as np
import tensorflow as tf
import pandas as pd


CENSUS_TRAIN_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)
CENSUS_TEST_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
)
COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


census_train_path = tf.contrib.keras.utils.get_file("census.train", CENSUS_TRAIN_URL)
census_test_path = tf.contrib.keras.utils.get_file("census.test", CENSUS_TEST_URL)
census_train = pd.read_csv(census_train_path, index_col=False, names=COLUMN_NAMES) 
census_test = pd.read_csv(census_test_path, skiprows=1, index_col=False, names=COLUMN_NAMES) 

print(census_train.head(10))
