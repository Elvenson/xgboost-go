package activation

import (
	"fmt"

	"github.com/Elvenson/xgboost-go/mat"
	"github.com/Elvenson/xgboost-go/protobuf"
	"github.com/chewxy/math32"
)

// Logistic is struct contains necessary data for doing logistic calculation
// for now is empty.
type Logistic struct{}

// sigmoid applies sigmoid transformation to value
func sigmoid(x float32) float32 {
	return 1.0 / (1.0 + math32.Exp(-x))
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
