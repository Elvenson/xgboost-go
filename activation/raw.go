package activation

import (
	"fmt"

	"github.com/Elvenson/xgboost-go/mat"
	"github.com/Elvenson/xgboost-go/protobuf"
)

// Raw is struct contains necessary data for doing logistic calculation
// for now is empty.
type Raw struct{}

// Transform does nothing just returns the raw prediction.
func (a *Raw) Transform(rawPredictions mat.Vector) (mat.Vector, error) {
	if len(rawPredictions) == 0 {
		return mat.Vector{}, fmt.Errorf("prediction should have at least 1 dimension")
	}
	return rawPredictions, nil
}

// Type returns activate type.
func (a *Raw) Type() protobuf.ActivateType {
	return protobuf.ActivateType_RAW
}

// Name returns activation name.
func (a *Raw) Name() string {
	return protobuf.ActivateType_name[int32(protobuf.ActivateType_RAW)]
}
