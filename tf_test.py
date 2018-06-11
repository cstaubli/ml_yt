import numpy as np 
import tensorflow as tf
from sklearn import cross_validation
from tensorflow.contrib import learn

iris = learn.datasets.load_dataset("iris")
x_train, x_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

