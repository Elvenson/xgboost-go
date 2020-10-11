package models

import (
	"path/filepath"
	"testing"

	"github.com/baobui/xgboost-go/activation"
)

func TestXGBoostJSONIris(t *testing.T) {
	modelPath := filepath.Join("../test/data", "iris_xgboost_dump.json")
	//featurePath := filepath.Join("testdata", "fmap_pandas.txt")
	_, err := LoadXGBoostFromJSON(modelPath, "", 3, 4,  &activation.Softmax{})
	if err != nil {
		t.Error(err)
	}
}