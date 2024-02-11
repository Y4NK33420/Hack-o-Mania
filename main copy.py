from joblib import dump, load
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
malData = pd.read_csv('MalwareData.csv', sep = "|")
# print(malData.head())
legit = malData[0:41323].drop(['legitimate'], axis=1)
mal = malData[41323::].drop(['legitimate'], axis=1)

# print(legit.head())
# print(mal.head())
# print(malData.columns)


#dropping the Name, MD5 and legitimate columns
data_in = malData.drop(['Name', 'md5', 'legitimate'], axis=1).values
labels = malData['legitimate'].values
extratrees = ExtraTreesClassifier().fit(data_in, labels)
select = SelectFromModel(extratrees, prefit=True)
data_in_new = select.transform(data_in)
print(data_in.shape, data_in_new.shape)

features = data_in_new.shape[1]
index = np.argsort(extratrees.feature_importances_)[::-1][:features]
for f in range(features):
    print("%d. feature %s (%f)" % (f + 1, malData.columns[2+index[f]], extratrees.feature_importances_[index[f]]))


# Splitting the data into training and testing sets
legit_train, legit_test, mal_train, mal_test = cross_validate.train_test_split(data_in_new, labels, test_size=0.2)
classif = RandomForestClassifier(n_estimators=50)

classif.fit(legit_train, mal_train)

dump(classif, 'malware.joblib')

load('malware.joblib')


