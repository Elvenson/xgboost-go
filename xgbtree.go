package xgboost

import (
	"fmt"

	"github.com/Elvenson/xgboost-go/mat"
)

// xgbtree constant values.
const (
	isLeaf = 1
)

type xgbNode struct {
	NodeID     int
	Threshold  float32
	Yes        int
	No         int
	Missing    int
	Feature    int
	Flags      uint8
	LeafValues float32
}

type xgbTree struct {
	nodes []*xgbNode
}

func (t *xgbTree) predict(features mat.SparseVector) (float32, error) {
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
