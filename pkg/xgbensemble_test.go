package pkg

import (
	"path/filepath"
	"testing"
)

func TestXGBoostJSONIris(t *testing.T) {
	modelPath := filepath.Join("testdata", "iris_xgboost_dump.json")
	//featurePath := filepath.Join("testdata", "fmap_pandas.txt")
	// TODO: Should pass transformation function inside.
	_, err := LoadXGBoostFromJSON(modelPath, "", 4,  false)
	if err != nil {
		t.Error(err)
	}
}