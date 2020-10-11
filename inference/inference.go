package inference

import (
	"github.com/baobui/xgboost-go/activation"
	"github.com/baobui/xgboost-go/mat"
)

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
