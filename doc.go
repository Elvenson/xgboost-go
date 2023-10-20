/*
Package xgboost is a pure Golang implementation of loading DMLC XGBoost json model generated from dump_model python
API. This package supports binary, multiclass and regression inference. Note that this package is just for inference
purpose only, for training part please reference to https://github.com/dmlc/xgboost.

# Training model

In order to have a json encoded model file, we need to train the model via Python first:

iris_xgboost.py:

	import xgboost as xgb
	from sklearn import datasets
	from sklearn.model_selection import train_test_split
	from sklearn.datasets import dump_svmlight_file
	import numpy as np

	X, y = datasets.load_iris(return_X_y=True)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

	dtrain = xgb.DMatrix(X_train, label=y_train)
	param = {'max_depth': 4, 'eta': 1, 'objective': 'multi:softmax', 'nthread': 4,
			 'eval_metric': 'auc', 'num_class': 3}

	num_round = 10
	bst = xgb.train(param, dtrain, num_round)
	y_pred = bst.predict(xgb.DMatrix(X_test))

	clf = xgb.XGBClassifier(max_depth=4, objective='multi:softprob', n_estimators=10,
							num_classes=3)

	clf.fit(X_train, y_train)
	y_pred_proba = clf.predict_proba(X_test)

	np.savetxt('../data/iris_xgboost_true_prediction.txt', y_pred, delimiter='\t')
	np.savetxt('../data/iris_xgboost_true_prediction_proba.txt', y_pred_proba, delimiter='\t')
	dump_svmlight_file(X_test, y_test, '../data/iris_test.libsvm')
	bst.dump_model('../data/iris_xgboost_dump.json', dump_format='json')

Here is how to load the model exported from the above script:

	package main

	import (
		"fmt"

		"github.com/Elvenson/xgboost-go/activation"
		"github.com/Elvenson/xgboost-go/mat"
		"github.com/Elvenson/xgboost-go/models"
	)

	func main() {
		ensemble, err := models.LoadXGBoostFromJSON("your model path",
			"", 1, 4, &activation.Logistic{})
		if err != nil {
			panic(err)
		}

		input, err := mat.ReadLibsvmFileToSparseMatrix("your libsvm input path")
		if err != nil {
			panic(err)
		}
		predictions, err := ensemble.PredictProba(input)
		if err != nil {
			panic(err)
		}
		fmt.Printf("%+v\n", predictions)
	}

For more information, please take a look at xgbensemble_test.go
*/
package xgboost
