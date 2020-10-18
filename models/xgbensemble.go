package models

import "github.com/baobui/xgboost-go/mat"

type xgbEnsemble struct {
	Trees      []*xgbTree
	name       string
	numClasses int
	numFeat    int
}

// Name returns name of ensemble model.
func (e *xgbEnsemble) Name() string {
	return e.name
}

// NumFeatures returns number of features for this ensemble model.
func (e *xgbEnsemble) NumFeatures() int {
	return e.numFeat
}

// PredictInner returns prediction of this ensemble model.
func (e *xgbEnsemble) PredictInner(features mat.SparseVector) (mat.Vector, error) {
	// number of trees for 1 class.
	pred := make([]float64, e.numClasses)
	numTreesPerClass := len(e.Trees) / e.numClasses
	for i := 0; i < e.numClasses; i++ {
		for k := 0; k < numTreesPerClass; k++ {
			p, err := e.Trees[k*e.numClasses+i].predict(features)
			if err != nil {
				return mat.Vector{}, nil
			}
			pred[i] += p
		}
	}
	return pred, nil
}
