package activation

import (
	"github.com/baobui/xgboost-go/mat"
	"github.com/baobui/xgboost-go/protobuf"
)

// Activation ...
type Activation interface {
	Transform(rawPrediction mat.Vector) (mat.Vector, error)
	Type() protobuf.ActivateType
	Name() string
}
