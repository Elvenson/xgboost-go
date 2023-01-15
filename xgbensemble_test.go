package xgboost

import (
	"testing"

	"gotest.tools/assert"

	"github.com/Elvenson/xgboost-go/activation"
	"github.com/Elvenson/xgboost-go/mat"
)

func TestEnsemble_PredictBreastCancer(t *testing.T) {
	modelPath := "test/data/breast_cancer_xgboost_dump.json"
	ensemble, err := LoadXGBoostFromJSON(modelPath,
		"", 1, 4, &activation.Logistic{})
	assert.NilError(t, err)

	inputPath := "test/data/breast_cancer_test.libsvm"
	input, err := mat.ReadLibsvmFileToSparseMatrix(inputPath)
	assert.NilError(t, err)

	predictions, err := ensemble.PredictProba(input)
	assert.NilError(t, err)

	expectedPredPath := "test/data/breast_cancer_xgboost_true_prediction.txt"
	expectedClasses, err := mat.ReadCSVFileToDenseMatrix(expectedPredPath, "\t", 0.0)
	assert.NilError(t, err)

	err = mat.IsEqualMatrices(&predictions, &expectedClasses, 0.0001)
	assert.NilError(t, err)

	predictions, err = ensemble.Predict(input)
	assert.NilError(t, err)

	err = mat.IsEqualMatrices(&predictions, &expectedClasses, 0.0001)
	assert.NilError(t, err)

	// with undefined depth
	ensemble, err = LoadXGBoostFromJSON(modelPath,
		"", 1, 0, &activation.Logistic{})
	assert.NilError(t, err)

	predictions, err = ensemble.PredictProba(input)
	assert.NilError(t, err)

	err = mat.IsEqualMatrices(&predictions, &expectedClasses, 0.0001)
	assert.NilError(t, err)

	predictions, err = ensemble.Predict(input)
	assert.NilError(t, err)

	err = mat.IsEqualMatrices(&predictions, &expectedClasses, 0.0001)
	assert.NilError(t, err)

	assert.Check(t, len(ensemble.Name()) != 0)
}

func TestEnsemble_PredictBreastCancerFeatureMap(t *testing.T) {
	modelPath := "test/data/breast_cancer_xgboost_dump_fmap.json"
	ensemble, err := LoadXGBoostFromJSON(modelPath,
		"test/data/breast_cancer_fmap.txt", 1, 4, &activation.Logistic{})
	assert.NilError(t, err)

	inputPath := "test/data/breast_cancer_test.libsvm"
	input, err := mat.ReadLibsvmFileToSparseMatrix(inputPath)
	assert.NilError(t, err)

	predictions, err := ensemble.PredictProba(input)
	assert.NilError(t, err)

	expectedPredPath := "test/data/breast_cancer_xgboost_true_prediction.txt"
	expectedClasses, err := mat.ReadCSVFileToDenseMatrix(expectedPredPath, "\t", 0.0)
	assert.NilError(t, err)

	err = mat.IsEqualMatrices(&predictions, &expectedClasses, 0.0001)
	assert.NilError(t, err)

	predictions, err = ensemble.Predict(input)
	assert.NilError(t, err)

	err = mat.IsEqualMatrices(&predictions, &expectedClasses, 0.0001)
	assert.NilError(t, err)
}

func TestEnsemble_BreastCancerRegression(t *testing.T) {
	modelPath := "test/data/breast_cancer_xgboost_dump_regression.json"
	ensemble, err := LoadXGBoostFromJSON(modelPath,
		"", 1, 4, &activation.Raw{})
	assert.NilError(t, err)

	inputPath := "test/data/breast_cancer_test.libsvm"
	input, err := mat.ReadLibsvmFileToSparseMatrix(inputPath)
	assert.NilError(t, err)

	// base value is the target average value, check test/scripts/breast_cancer_xgboost.py for more detail.
	predictions, err := ensemble.PredictRegression(input, 0.6373626373626373)
	assert.NilError(t, err)

	expectedPredPath := "test/data/breast_cancer_xgboost_true_prediction_regression.txt"
	expectedClasses, err := mat.ReadCSVFileToDenseMatrix(expectedPredPath, "\t", 0.0)
	assert.NilError(t, err)

	err = mat.IsEqualMatrices(&predictions, &expectedClasses, 0.0001)
	assert.NilError(t, err)
}

func TestEnsemble_Iris(t *testing.T) {
	modelPath := "test/data/iris_xgboost_dump.json"
	ensemble, err := LoadXGBoostFromJSON(modelPath,
		"", 3, 4, &activation.Softmax{})
	assert.NilError(t, err)

	inputPath := "test/data/iris_test.libsvm"
	input, err := mat.ReadLibsvmFileToSparseMatrix(inputPath)
	assert.NilError(t, err)

	predictions, err := ensemble.Predict(input)
	assert.NilError(t, err)

	expectedClassesPath := "test/data/iris_xgboost_true_prediction.txt"
	expectedClasses, err := mat.ReadCSVFileToDenseMatrix(expectedClassesPath, "\t", 0.0)
	assert.NilError(t, err)

	err = mat.IsEqualMatrices(&predictions, &expectedClasses, 0.0000)
	assert.NilError(t, err)

	expectedProbPath := "test/data/iris_xgboost_true_prediction_proba.txt"
	expectedProb, err := mat.ReadCSVFileToDenseMatrix(expectedProbPath, "\t", 0.0)
	assert.NilError(t, err)

	predictions, err = ensemble.PredictProba(input)
	assert.NilError(t, err)

	err = mat.IsEqualMatrices(&predictions, &expectedProb, 0.0001)
	assert.NilError(t, err)

	// with undefined depth
	ensemble, err = LoadXGBoostFromJSON(modelPath,
		"", 3, 0, &activation.Softmax{})
	assert.NilError(t, err)

	predictions, err = ensemble.Predict(input)
	assert.NilError(t, err)

	err = mat.IsEqualMatrices(&predictions, &expectedClasses, 0.0000)
	assert.NilError(t, err)

	predictions, err = ensemble.PredictProba(input)
	assert.NilError(t, err)

	err = mat.IsEqualMatrices(&predictions, &expectedProb, 0.0001)
	assert.NilError(t, err)
}
