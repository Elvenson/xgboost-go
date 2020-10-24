package inference

import (
	"fmt"

	"github.com/baobui/xgboost-go/activation"
	"github.com/baobui/xgboost-go/mat"
	"github.com/baobui/xgboost-go/protobuf"
)

// EnsembleBase ...
type EnsembleBase interface {
	PredictInner(features mat.SparseVector) (mat.Vector, error)
	Name() string
	NumClasses() int
}

// EnsembleBase ...
type Ensemble struct {
	EnsembleBase
	activation.Activation
}

func (e *Ensemble) PredictRegression(features mat.SparseMatrix, baseVal float64) (mat.Matrix, error) {
	if e.NumClasses() == 0 {
		return mat.Matrix{}, fmt.Errorf("0 class please check your model")
	}
	if e.NumClasses() != 1 {
		return mat.Matrix{}, fmt.Errorf("regression prediction only support binary classes for now")
	}

	results := mat.Matrix{Vectors: make([]*mat.Vector, len(features.Vectors))}
	for i, row := range features.Vectors {
		pred, err := e.PredictInner(row)
		if err != nil {
			return mat.Matrix{}, err
		}
		if len(pred) != e.NumClasses() {
			return mat.Matrix{}, fmt.Errorf("number of predicted value (%d) must match number of classes (%d)",
				len(pred), e.NumClasses())
		}
		if e.Type() != protobuf.ActivateType_RAW {
			return mat.Matrix{}, fmt.Errorf("regression model must have raw activation")
		}
		pred, err = e.Transform(pred)
		pred[0] += baseVal
		if err != nil {
			return mat.Matrix{}, err
		}
		results.Vectors[i] = &pred
	}
	return results, nil
}

// PredictProba predicts probabilities using ensemble model interface.
func (e *Ensemble) PredictProba(features mat.SparseMatrix) (mat.Matrix, error) {
	if e.NumClasses() == 0 {
		return mat.Matrix{}, fmt.Errorf("0 class please check your model")
	}

	results := mat.Matrix{Vectors: make([]*mat.Vector, len(features.Vectors))}
	for i, row := range features.Vectors {
		pred, err := e.PredictInner(row)
		if err != nil {
			return mat.Matrix{}, err
		}
		if len(pred) != e.NumClasses() {
			return mat.Matrix{}, fmt.Errorf("number of predicted value (%d) must match number of classes (%d)",
				len(pred), e.NumClasses())
		}
		pred, err = e.Transform(pred)
		if err != nil {
			return mat.Matrix{}, err
		}
		results.Vectors[i] = &pred
	}
	return results, nil
}

// Predict predicts class using ensemble model interface.
func (e *Ensemble) Predict(features mat.SparseMatrix) (mat.Matrix, error) {
	if e.NumClasses() == 0 {
		return mat.Matrix{}, fmt.Errorf("0 class please check your model")
	}
	results := mat.Matrix{Vectors: make([]*mat.Vector, len(features.Vectors))}
	for i, row := range features.Vectors {
		pred, err := e.PredictInner(row)
		if err != nil {
			return mat.Matrix{}, err
		}
		if len(pred) != e.NumClasses() {
			return mat.Matrix{}, fmt.Errorf("number of predicted value (%d) must match number of classes (%d)",
				len(pred), e.NumClasses())
		}
		if len(pred) == 0 {
			return mat.Matrix{}, fmt.Errorf("empty inner prediction")
		}
		pred, err = e.Transform(pred)
		if err != nil {
			return mat.Matrix{}, err
		}
		if e.NumClasses() == 1 {
			// for binary classification prediction results is probabilities.
			results.Vectors[i] = &pred
		} else {
			idx, err := mat.GetVectorMaxIdx(&pred)
			if err != nil {
				return mat.Matrix{}, err
			}
			results.Vectors[i] = &mat.Vector{float64(idx)}
		}
	}

	return results, nil
}
