package models

import (
	"log"
	"math"
)

// xgbtree constant values.
const (
	isLeaf        = 1
	zeroThreshold = 1e-35
)

type xgbNode struct {
	NodeID     int
	Threshold  float64
	Yes        int
	No         int
	Missing    int
	Feature    int
	Flags      uint8
	LeafValues float64
}

type xgbTree struct {
	nodes []*xgbNode
}

func (t *xgbTree) predict(features []float64) float64 {
	idx := 0
	for {
		node := t.nodes[idx]
		if node == nil {
			log.Fatalf("nil node")
		}
		if node.Feature >= len(features) {
			log.Fatalf("wrong input dimension: %d, please check your input data", len(features))
		}
		if node.Flags&isLeaf > 0 {
			return node.LeafValues
		}
		v := features[node.Feature]
		if math.IsNaN(v) {
			// missing value will be represented as NaN value.
			idx = node.Missing
		} else if v >= node.Threshold {
			idx = node.No
		} else {
			idx = node.Yes
		}
	}
}
