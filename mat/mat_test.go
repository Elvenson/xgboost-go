package mat

import (
	"fmt"
	"testing"

	"gotest.tools/assert"
)

func TestReadLibsvmFile(t *testing.T) {
	m, err := ReadLibsvmFile("../test/data/iris_test.libsvm")
	assert.NilError(t, err)
	fmt.Printf("%+v", m)
}
