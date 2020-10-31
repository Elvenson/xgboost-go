package mat

import (
	"testing"

	"gotest.tools/assert"
)

func TestReadLibsvmFile(t *testing.T) {
	m, err := ReadLibsvmFileToSparseMatrix("../test/data/iris_test.libsvm")
	assert.NilError(t, err)
	assert.Check(t, len(m.Vectors) != 0)
	assert.Equal(t, len(m.Vectors[0]), 4)
}

func TestReadCSVFileToDenseMatrix(t *testing.T) {
	m, err := ReadCSVFileToDenseMatrix(
		"../test/data/iris_xgboost_true_prediction_proba.txt", "\t", 0)
	assert.NilError(t, err)
	assert.Check(t, len(m.Vectors) != 0)
	assert.Equal(t, len(*m.Vectors[0]), 3)
}
