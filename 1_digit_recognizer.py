import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding


###############################		Digit Recognizer 	###############################
#######################################################################################

# 	Loading data

sub     = pd.read_csv('cnn_mnist_d.csv')
sub_2   = pd.read_csv('data/sample_submission.csv')

print('\n\t\t\t Start Off \n')
print(sub.shape)
print(sub_2.shape)
print('\n')
print(type(sub))
print(type(sub_2))

