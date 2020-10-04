package pkg

type xgbNode struct {
	Threshold float64
	Left uint32
	Right uint32
	Feature uint32
	Flags uint8
}

type xgbTree struct {
	nodes []*xgbNode
	leafValues []float64
}