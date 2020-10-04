package pkg

type xgbEnsemble struct {
	Trees      []*xgbTree
	numClasses int
	name       string
}