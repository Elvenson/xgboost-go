import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import xgboost as xgb
from sklearn import datasets
from sklearn.datasets import dump_svmlight_file

# get data
diamonds = pd.read_csv("../data/diamonds.csv")
diamonds.price = 1 * (diamonds.price > 3000)

cut = pd.get_dummies(diamonds.color, drop_first=True)
cut.columns = [f"cut_{c}" for c in cut.columns.tolist()]
diamonds = diamonds.drop('cut', axis = 1)
diamonds = pd.concat([diamonds, cut], axis=1)

clarity = pd.get_dummies(diamonds.clarity, drop_first=True)
clarity.columns = [f"clarity_{c}" for c in clarity.columns.tolist()]
diamonds = diamonds.drop('clarity', axis = 1)
diamonds = pd.concat([diamonds, clarity], axis=1)

color = pd.get_dummies(diamonds.color, drop_first=True)
color.columns = [f"color{c}" for c in color.columns.tolist()]
diamonds = diamonds.drop('color', axis = 1)
diamonds = pd.concat([diamonds, color], axis=1)

X, y = diamonds.drop('price', axis=1), diamonds[['price']]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# training
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    n_jobs=-1, 
    verbose_eval=None,
)

model.fit(X_train.values, y_train.values.ravel())

# inference
bst = model.get_booster()
y_pred_probas = model.predict_proba(X_test.values)[:,1]
y_pred = model.predict(X_test.values)

# save model
np.savetxt('../data/diamonds_xgboost_true_prediction_proba.txt', y_pred_probas, delimiter='\t')
dump_svmlight_file(X_test.values, y_test.values.ravel(), '../data/diamonds_test.libsvm')
bst.dump_model('../data/diamonds_xgboost_dump.json', dump_format='json')