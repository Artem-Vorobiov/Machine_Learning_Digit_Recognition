import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

train   = pd.read_csv('data/train.csv')
Y_train = train['label']
X_train = train.drop(labels = ["label"],axis = 1) 

print('\n\t\t\t Start Off \n')
print(train.shape)
print('\n\n')


# sns.countplot(Y_train)
# plt.show()

###################################################
###### 			КРУТАЯ ШТУКА 		######

#	КАК ЭТО МОЖНО СДЕЛАТЬ В РУЧНУЮ ?
Y_train = to_categorical(Y_train, num_classes = 10)
###################################################

print(Y_train)
print('\n\n')
print(Y_train.shape)