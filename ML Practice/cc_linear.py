import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mserr 


path = "E:\\MSIT-II\\Data Science Specialization\\ML Practice\\"
train = path + "train.csv"
test = path + "test.csv"

trainx_df = pd.read_csv(train, index_col = 'Id')
# print(trainx_df.shape)
trainy_df = trainx_df['SalePrice']
# print(trainy_df.shape)

trainx_df.drop('SalePrice', axis = 1, inplace = True)
# print(trainx_df.shape)

testx_df = pd.read_csv(test, index_col = 'Id')
# print(testx_df.shape)

# print(trainx_df.isnull().sum())

sample_size = len(trainx_df)
# print(sample_size)

columns_with_null_values = [[col, float(trainx_df[col].isnull().sum()) / float(sample_size)] for col in trainx_df.columns if trainx_df[col].isnull().sum()]
# print(columns_with_null_values)

# for i in range(len(columns_with_null_values)):
    # print(columns_with_null_values[i])

columns_to_drop = [x for [x, y] in columns_with_null_values if y > 0.3]
# print(columns_to_drop)

trainx_df.drop(columns_to_drop, axis = 1, inplace = True)
testx_df.drop(columns_to_drop, axis = 1, inplace = True)

# print(trainx_df.shape)
# print(testx_df.shape)

# trainx_df.dropna(axis = 0, inplace = True)
# print(trainx_df.shape)


categorical_columns = [col for col in trainx_df.columns if
trainx_df[col].dtype == object]
# categorical_columns.append('MSSubClass')
# print(categorical_columns)
# print(len(list(categorical_columns)))

ordinal_columns = [col for col in trainx_df.columns if col not in categorical_columns]
# print(ordinal_columns)
# print(len(list(ordinal_columns)))


dummy_row = list()
for col in trainx_df.columns:
    if col in categorical_columns:
        dummy_row.append("dummy")
    else:
        dummy_row.append("")

# print(dummy_row)

new_row = pd.DataFrame([dummy_row], columns = trainx_df.columns)
trainx_df = pd.concat([trainx_df, new_row], axis = 0, ignore_index = True)
testx_df = pd.concat([testx_df], axis = 0, ignore_index = True)

# trainx_df.to_csv("dummy.csv")

for col in categorical_columns:
    trainx_df[col].fillna(value = "dummy", inplace = True)
    testx_df[col].fillna(value = "dummy", inplace = True)



enc = OneHotEncoder(drop = 'first', sparse = False)
enc.fit(trainx_df[categorical_columns])
# print(enc.get_feature_names(categorical_columns))
trainx_enc = pd.DataFrame(enc.transform(trainx_df[categorical_columns]))
testx_enc = pd.DataFrame(enc.transform(testx_df[categorical_columns]))
trainx_enc.columns = enc.get_feature_names(categorical_columns)
testx_enc.columns = enc.get_feature_names(categorical_columns)

trainx_df = pd.concat([trainx_df[ordinal_columns], trainx_enc], axis = 1, ignore_index = True)
testx_df = pd.concat([testx_df[ordinal_columns], testx_enc], axis = 1, ignore_index = True)

# trainx_df.to_csv("encodedTrainx.csv")


trainx_df.drop(trainx_df.tail(1).index, inplace = True)


imputer = KNNImputer(n_neighbors = 2)
imputer.fit(trainx_df)
trainx_df_filled = imputer.transform(trainx_df)
trainx_df_filled = pd.DataFrame(trainx_df_filled, columns = trainx_df.columns)
trainx_df_filled.reset_index(drop = True, inplace = True)

# print(trainx_df_filled.isnull().sum())

testx_df_filled = imputer.transform(testx_df)
testx_df_filled = pd.DataFrame(testx_df_filled, columns = testx_df.columns)
testx_df_filled.reset_index(drop = True, inplace = True)


scaler = preprocessing.StandardScaler().fit(trainx_df_filled)

trainx_df_filled = scaler.transform(trainx_df_filled)
testx_df_filled = scaler.transform(testx_df_filled)




X_train, X_test, y_train, y_test = 
train_test_split(trainx_df_filled, trainy_df.values.rave())




