import xgboost as xgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import dump_svmlight_file
import numpy as np

X, y = datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# For binary classification.
dtrain = xgb.DMatrix(X_train, label=y_train)
classification_param = {'max_depth': 4, 'eta': 1, 'objective': 'binary:logistic', 'nthread': 4, 'eval_metric': 'auc'}

num_round = 10
bst = xgb.train(classification_param, dtrain, num_round)
y_pred = bst.predict(xgb.DMatrix(X_test))

np.savetxt('../data/breast_cancer_xgboost_true_prediction.txt', y_pred, delimiter='\t')
dump_svmlight_file(X_test, y_test, '../data/breast_cancer_test.libsvm')
bst.dump_model('../data/breast_cancer_xgboost_dump.json', dump_format='json')

# Dump model with feature map.
bst.dump_model('../data/breast_cancer_xgboost_dump_fmap.json', dump_format='json', fmap='../data/breast_cancer_fmap.txt')

# For regression.
y_train_mean = np.mean(y_train)
regression_params = {
    'base_score': y_train_mean,
    'max_depth': 4,
    'eta': 1,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'silent': 1
}
print("mean y_train is: {}".format(y_train_mean))

bst = xgb.train(params=regression_params, dtrain=dtrain, num_boost_round=num_round)
y_pred = bst.predict(xgb.DMatrix(X_test))
np.savetxt('../data/breast_cancer_xgboost_true_prediction_regression.txt', y_pred, delimiter='\t')
bst.dump_model('../data/breast_cancer_xgboost_dump_regression.json', dump_format='json')

