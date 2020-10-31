package models

import (
	"fmt"

	"github.com/baobui/xgboost-go/mat"
)

// xgbtree constant values.
const (
	isLeaf = 1
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

func (t *xgbTree) predict(features mat.SparseVector) (float64, error) {
	idx := 0
	for {
		node := t.nodes[idx]
		if node == nil {
			return 0, fmt.Errorf("nil node")
		}
		if node.Flags&isLeaf > 0 {
			return node.LeafValues, nil
		}
		v, ok := features[node.Feature]
		if !ok {
			// missing value will be represented as NaN value.
			idx = node.Missing
		} else if v >= node.Threshold {
			idx = node.No
		} else {
			idx = node.Yes
		}
	}
}
