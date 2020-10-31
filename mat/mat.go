package mat

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/pkg/errors"
)

// Vector is a list of float numbers.
type Vector []float64

// SparseVector is a map with index is a key and value is a value at that index.
type SparseVector map[int]float64

// SparseMatrix is a list of sparse vectors.
type SparseMatrix struct {
	Vectors []SparseVector
}

// Matrix is a list of vector.
type Matrix struct {
	Vectors []*Vector
}

// ReadLibsvmFileToSparseMatrix reads libsvm file into sparse matrix.
func ReadLibsvmFileToSparseMatrix(fileName string) (SparseMatrix, error) {
	file, err := os.Open(fileName)
	if err != nil {
		return SparseMatrix{}, fmt.Errorf("unable to open %s: %s", fileName, err)
	}
	defer file.Close()

	reader := bufio.NewReader(file)

	sparseMatrix := SparseMatrix{Vectors: make([]SparseVector, 0)}
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err != io.EOF {
				return SparseMatrix{}, err
			}
			break

		}
		line = strings.TrimSpace(line)
		if line == "" {
			break
		}
		tokens := strings.Split(line, " ")
		if len(tokens) < 2 {
			return SparseMatrix{}, fmt.Errorf("too few columns")
		}
		// first column is label so skip it.
		vec := SparseVector{}
		for c := 1; c < len(tokens); c++ {
			if len(tokens[c]) == 0 {
				return SparseMatrix{}, fmt.Errorf("corrupted data format please check for empty spaces")
			}
			pair := strings.Split(tokens[c], ":")
			if len(pair) != 2 {
				return SparseMatrix{}, fmt.Errorf("wrong data format %s", tokens[c])
			}
			colIdx, err := strconv.ParseUint(pair[0], 10, 32)
			if err != nil {
				return SparseMatrix{}, fmt.Errorf("cannot parse to int %s: %s", pair[0], err)
			}
			val, err := strconv.ParseFloat(pair[1], 64)
			if err != nil {
				return SparseMatrix{}, fmt.Errorf("cannot parse to float %s: %s", pair[1], err)
			}
			vec[int(colIdx)] = val
		}
		sparseMatrix.Vectors = append(sparseMatrix.Vectors, vec)
	}
	return sparseMatrix, nil
}

// ReadCSVFileToDenseMatrix reads CSV file to dense matrix.
func ReadCSVFileToDenseMatrix(fileName string, delimiter string, defaultVal float64) (Matrix, error) {
	file, err := os.Open(fileName)
	if err != nil {
		return Matrix{}, fmt.Errorf("unable to open %s: %s", fileName, err)
	}
	defer file.Close()

	reader := bufio.NewReader(file)

	matrix := Matrix{Vectors: make([]*Vector, 0)}
	colDim := -1
	row := 0
	for {
		line, err := reader.ReadString('\n')
		if err != nil && err != io.EOF {
			return Matrix{}, err
		}
		line = strings.TrimSpace(line)
		if line == "" {
			break
		}
		tokens := strings.Split(line, delimiter)
		vec := Vector{}
		for i := 0; i < len(tokens); i++ {
			var val float64
			if len(tokens[i]) == 0 {
				val = defaultVal
			} else {
				v, err := strconv.ParseFloat(tokens[i], 64)
				if err != nil {
					return Matrix{}, fmt.Errorf("cannot convert to float %s: %s", tokens[i], err)
				}
				val = v
			}
			vec = append(vec, val)
		}
		if colDim == -1 {
			colDim = len(vec)
		} else if colDim != len(vec) {
			return Matrix{}, fmt.Errorf("row %d has different dimension: %d, please check your file",
				row, len(vec))
		}
		matrix.Vectors = append(matrix.Vectors, &vec)
		row++
	}
	return matrix, nil
}

// IsEqualVectors compares 2 vectors with a threshold.
func IsEqualVectors(v1, v2 *Vector, threshold float64) error {
	if len(*v1) != len(*v2) {
		return fmt.Errorf("different vector length v1=%d, v2=%d", len(*v1), len(*v2))
	}
	for i := range *v1 {
		if math.Abs((*v1)[i]-(*v2)[i]) > threshold {
			return fmt.Errorf("%d element mismatch: v1[%d]=%f, v2[%d]=%f", i, i, (*v1)[i], i, (*v2)[i])
		}
	}
	return nil
}

// GetVectorMaxIdx gets the index of the maximum value within a vector.
func GetVectorMaxIdx(v *Vector) (int, error) {
	if len(*v) == 0 {
		return -1, fmt.Errorf("empty vector")
	}
	maxVal := math.Inf(-1)
	r := 0
	for idx, i := range *v {
		if i > maxVal {
			maxVal = i
			r = idx
		}
	}
	return r, nil
}

// IsEqualMatrices compares 2 matrices with a threshold.
func IsEqualMatrices(m1, m2 *Matrix, threshold float64) error {
	if len(m1.Vectors) != len(m2.Vectors) {
		return fmt.Errorf("row  matrix mismatch: m1 got %d rows, m2 got %d rows", len(m1.Vectors), len(m2.Vectors))
	}
	for i := range m1.Vectors {
		err := IsEqualVectors(m1.Vectors[i], m2.Vectors[i], threshold)
		if err != nil {
			return errors.Wrap(err, fmt.Sprintf("matrix comparison at index %d", i))
		}
	}
	return nil
}
