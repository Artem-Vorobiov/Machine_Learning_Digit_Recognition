import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding


###############################		Digit Recognizer 	###############################
#######################################################################################

# 	Loading data

train   = pd.read_csv('data/train.csv')
Y_train = train['label']
X_train = train.drop(labels = ["label"],axis = 1) 
print('\n\t\t\t Start Off \n')
# print(train.shape)
# print('\n\n')


#	Plot data

# sns.countplot(Y_train)
# plt.show()


#	Categorization
###################################################
###### 			КРУТАЯ ШТУКА 		######

#	КАК ЭТО МОЖНО СДЕЛАТЬ В РУЧНУЮ ?
Y_train = to_categorical(Y_train, num_classes = 10)
###################################################
X_train   = X_train.values.reshape(-1,28,28,1)



#	Visualize Digits
# print(Y_train)
# print('\n\n')
# print(Y_train.shape)
# print(X_train[0].shape)
# print(X_train[0])
# pic_num   = X_train.values.reshape(-1,28,28)
# print(pic_num)
# print(pic_num.shape)
# plt.imshow(pic_num[0])
# plt.show()