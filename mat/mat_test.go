package mat

import (
	"fmt"
	"testing"

	"gotest.tools/assert"
)

// TODO: Add more test.
func TestReadLibsvmFile(t *testing.T) {
	m, err := ReadLibsvmFileToSparseMatrix("../test/data/iris_test.libsvm")
	assert.NilError(t, err)
	fmt.Printf("%+v\n", m)
}

func TestReadCSVFileToDenseMatrix(t *testing.T) {
	m, err := ReadCSVFileToDenseMatrix("../test/data/iris_test.tsv", "\t", 0)
	assert.NilError(t, err)
	for _, v := range m.Vectors{
		fmt.Printf("%+v\n", v)
	}
}