package mat

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// Vector ...
type Vector []float64

// SparseVector ...
type SparseVector map[int]float64

// SparseMatrix ...
type SparseMatrix struct {
	Vectors []SparseVector
}

// Matrix ...
type Matrix struct {
	Vectors []*Vector
}

// ReadLibsvmFile reads libsvm file into sparse matrix.
func ReadLibsvmFile(fileName string) (SparseMatrix, error) {
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
			} else {
				break
			}
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
				break
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
