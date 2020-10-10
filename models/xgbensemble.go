package models

type xgbEnsemble struct {
	Trees      []*xgbTree
	name       string
	numClasses int
	numFeat    int
}

