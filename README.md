# Arrhythmia-Classification

![Screenshot](https://github.com/justinbrinkman/Arrhythmia-Classification/blob/master/ekg.png)

This analysis was performed for a 24-hour hackathon.  Our goal was to identify and optimize the best
model for classifying EKG results into one of 6 categories in order to assist medical professionals 
in the detection and general classification of arrhythmia.  Other analyses have shown that high
accuracy scores can be achieved when performing binary classification (Normal vs Arrhythmia), but
not for multi-classification.

Data for this project was taken from the UCI Machine Learning Repository.
Data can be accessed here: https://archive.ics.uci.edu/ml/datasets/arrhythmia

EKG Image source: https://en.wikipedia.org/wiki/Electrocardiography

Required imports to run model.py:
import pandas as pd, 
import matplotlib.pyplot as plt, 
import numpy as np, 
from sklearn.ensemble import RandomForestClassifier, 
from sklearn.metrics import accuracy_score, 
    

ANALYSIS

STEP 1) Data cleaning
1.	This data has many non-numerical values, so they were all converted to NaN
2.	Over half the cells in column 14 were NaN, so the column was dropped
3.	Remaining NaN values in the data set were filled by the calculated median values of their respective columns
4.	Reduce 16 categories to 6 by physiological similarity: “Normal”, “Rate”, “Rhythm”, “Infarction”, “Block” and “Other”

STEP 2) Classification by various scikit-learn stock models

![Screenshot](https://github.com/justinbrinkman/Arrhythmia-Classification/blob/master/stockmodelgraph.PNG)

STEP 3) Select best model to optimize: Random Forest

STEP 4) Optimize the model

The following techniques were used to try to improve the classification accuracy of the Random Forest model:
1.	Feature Selection – top features (of various levels) based on feature importance were used to generate new “selective” data frames for training the model.  This method resulted in no significant improvement of classification accuracy.
2.	Define Heart Rate Bounds -  it was noticed that the model had a hard time classifying instances as “Rate”, so bounds were set and tweaked to train the data.  E.g. all instances with heart rate values >98 or <56 were classified as “Rate”.  This method resulted in no significant improvement of classification accuracy.
3.	Ensembling – the Random Forest, SVC and KNeighbors models were trained and tested on the same data sets.  If two or more models classified an instance similarly, then that classification was selected.  If all models classified an instance differently, the Random Forest’s classification was chosen.  This method resulted in no significant improvement of classification accuracy.  This is likely due to the poor fitting of the SVC and KNeighbors models, even after tuning their respective parameters.
4.	Parameter Tuning – Random Forest model parameters were adjusted to achieve a better fit to the data.  This method improved the model accuracy from 71% to 77% when averaged over 100 trials.

Conclusion / Takeaways
1.	With such a small sample size and large number of features, it is difficult to achieve high classification accuracy for 6 classes by machine learning algorithms.  It is likely that the model is highly overfitting the data.
2.	It may be possible to engineer new features that can be easily interpreted by the algorithm.  For instance, ‘Rhythm’ was often misclassified by our model.  A likely strong feature would be standard deviations of time between beats.
3.	When in a hackathon, know when to stop tweaking the model to achieve miniscule improvements.  This took time away from experimenting with data visualizations.

