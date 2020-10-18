package inference

import (
	"github.com/baobui/xgboost-go/activation"
	"github.com/baobui/xgboost-go/mat"
)

// Ensemble ...
type Ensemble interface {
	PredictInner(features mat.SparseVector) (mat.Vector, error)
	Name() string
	NumFeatures() int
}

// EnsembleBase ...
type EnsembleBase struct {
	Ensemble
	activation.Activation
}

// Predict predicts using ensemble model interface.
func (e *EnsembleBase) Predict(features mat.SparseMatrix) (mat.Matrix, error) {
	results := mat.Matrix{Vectors: make([]*mat.Vector, len(features.Vectors))}
	for i, row := range features.Vectors {
		pred, err := e.PredictInner(row)
		if err != nil {
			return mat.Matrix{}, err
		}
		pred, err = e.Transform(pred)
		if err != nil {
			return mat.Matrix{}, err
		}
		results.Vectors[i] = &pred
	}
	return results, nil
}
