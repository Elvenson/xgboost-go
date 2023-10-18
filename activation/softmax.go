package activation

import (
	"fmt"

	"github.com/Elvenson/xgboost-go/mat"
	"github.com/Elvenson/xgboost-go/protobuf"
	"github.com/chewxy/math32"
)

// Softmax is struct contains necessary data for doing logistic calculation
// for now is empty.
type Softmax struct{}

// softmax function.
func softmax(vector mat.Vector) mat.Vector {
	var sum float32 = 0.0
	r := make([]float32, len(vector))
	for i, v := range vector {
		exp := math32.Exp(v)
		r[i] = exp
		sum += exp
	}
	if sum != 0.0 {
		inverseSum := 1.0 / sum
		for i := range r {
			r[i] *= inverseSum
		}
	}
	return r
}

// Transform passes prediction through softmax function.
func (a *Softmax) Transform(rawPredictions mat.Vector) (mat.Vector, error) {
	if len(rawPredictions) == 0 {
		return mat.Vector{}, fmt.Errorf("prediction should have at least 1 dimension")
	}

	p := softmax(rawPredictions)
	return p, nil
}

// Type returns activation type.
func (a *Softmax) Type() protobuf.ActivateType {
	return protobuf.ActivateType_SOFTMAX
}

// Name returns activation name.
func (a *Softmax) Name() string {
	return protobuf.ActivateType_name[int32(protobuf.ActivateType_SOFTMAX)]
}
