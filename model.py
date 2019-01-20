import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
##
## NOTE: this work was done in a Jupyter Notebook and only the necessary code has been
##      copied into this .py file.  This code imports the data and iterates 100x through
##      a loop that randomly splits the data into train/test sets and fits/predicts the
##      classification.  A value is returned which is the average accuracy score
##      over 100 trials.

## get the data
df = pd.read_csv('C:/Users/Justin/Desktop/Data Science/CornHacks/Arrhythmia/arrhythmia.csv')

## Convert everything to numeric. If it can't be, input NaN
## This is to remove unwanted characters as '?'s were noticed in the data
df = df.apply(pd.to_numeric, args=('coerce',))

## Reduce classes 1-16 to 6 categories:
## Normal, Infarction, Rhythm, Rate, Block, Other
## Create a dictionary to hold corresponding letters
classes = {1:'Normal', 2:'Infarction', 3:'Infarction', 4:'Infarction', 
           5:'Rate', 6:'Rate', 7:'Rhythm', 
           8:'Rhythm', 9:'Block', 10:'Block', 11:'Block', 
           12:'Block', 13:'Block', 14:'Other', 15:'Rhythm', 16:'Other'}
class_column = []

## Assign dictionary values and add to list class_column
for elem in df['280']:
    class_column.append(classes[elem])
    
## Replace column 280 (target column) with class_column list
df['280'] = class_column

##Drop column 14 due to too many NaNs
df = df.drop('14', axis=1)

## fill remaining NaNs with column median values. Values calculated in Excel
df['11'].fillna(41, inplace=True)
df['12'].fillna(56, inplace=True)
df['13'].fillna(40, inplace=True)
df['15'].fillna(72, inplace=True)

print("Computing...")

## Loop to train the model and predict the target.  Computes average accuracy score.
score = []
count = 0
while count < 100:
    ### START Random Forest Classification
    mskw = np.random.rand(len(df)) < 0.8
    train = df[mskw]
    test = df[~mskw]
    train_X = train.drop('280', axis=1)
    train_y = train['280']
    test_X = test.drop('280', axis=1)
    test_y = test['280']
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators = 100, max_depth = 110,min_samples_leaf = 1, max_features = 50)
    rfc.fit(train_X, train_y)
    predictions = rfc.predict(test_X)
    from sklearn.metrics import accuracy_score
    score.append(accuracy_score(test_y, predictions))
    count += 1

print("The average accuracy score of this model is: " + str(np.mean(score)))
