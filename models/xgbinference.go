package models

import "github.com/dmitryikh/leaves/transformation"

type ensemble interface {
	PredictInner(input, predictions []float64, startIndex int)
	Name() string
	NFeatures() int

}
type EnsembleBase struct {
	ensemble
	transformation.Transform
}
