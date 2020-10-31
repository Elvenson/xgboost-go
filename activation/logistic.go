package activation

import (
	"fmt"
	"math"

	"github.com/baobui/xgboost-go/mat"
	"github.com/baobui/xgboost-go/protobuf"
)

// Logistic is struct contains necessary data for doing logistic calculation
// for now is empty.
type Logistic struct{}

// sigmoid applies sigmoid transformation to value
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Transform passes prediction through logistic function.
func (a *Logistic) Transform(rawPredictions mat.Vector) (mat.Vector, error) {
	if len(rawPredictions) != 1 {
		return mat.Vector{}, fmt.Errorf("prediction should have only 1 dimension got %d", len(rawPredictions))
	}
	rawPredictions[0] = sigmoid(rawPredictions[0])
	return rawPredictions, nil
}

// Type returns activation type.
func (a *Logistic) Type() protobuf.ActivateType {
	return protobuf.ActivateType_LOGISTIC
}

// Name returns activation name.
func (a *Logistic) Name() string {
	return protobuf.ActivateType_name[int32(protobuf.ActivateType_LOGISTIC)]
}
