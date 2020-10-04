package pkg

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/dmitryikh/leaves/transformation"
)

type xgboostNode struct {
	NodeID                int     `json:"nodeid,omitempty"`
	SplitFeatureID        string  `json:"split,omitempty"`
	SplitFeatureThreshold float64 `json:"split_condition,omitempty"`
	YesID                 int     `json:"yes,omitempty"`
	NoID                  int     `json:"no,omitempty"`
	MissingID             int     `json:"missing,omitempty"`
	LeafValue             float64 `json:"leaf,omitempty"`
}

type xgboostJSON struct {
	xgboostNode
	Children []*xgboostNode `json:"children,omitempty"`
}

func loadFeatureMap(filePath string) (map[string]int, error) {
	featureFile, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer featureFile.Close()

	read := bufio.NewReader(featureFile)
	featureMap := make(map[string]int, 0)
	for {
		// feature map format: feature_index feature_name feature_type
		line, err := read.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, err
		}
		tk := strings.Split(line, " ")
		if len(tk) != 3 {
			return nil, fmt.Errorf("wrong feature map format")
		}
		featIdx, err := strconv.Atoi(tk[0])
		if err != nil {
			return nil, err
		}
		if _, ok := featureMap[tk[1]]; ok {
			return nil, fmt.Errorf("duplicate feature name")
		}
		featureMap[tk[1]] = featIdx
	}
	return featureMap, nil
}


func buildTree(xgbTreeJSON *xgboostJSON) (*xgbTree, error) {
	return nil, nil
}

// LoadXGBoostFromJSON loads xgboost model from json file.
func LoadXGBoostFromJSON(modelPath,
	featuresMapPath string,
	numClasses int,
	loadTransformation bool) (*EnsembleBase, error) {
	modelFile, err := os.Open(modelPath)
	if err != nil {
		return nil, err
	}
	defer modelFile.Close()

	var xgbEnsembleJSON []*xgboostJSON

	dec := json.NewDecoder(modelFile)
	err = dec.Decode(&xgbEnsembleJSON)
	if err != nil {
		return nil, err
	}
	var featMap map[string]int
	if len(featuresMapPath) != 0 {
		featMap, err = loadFeatureMap(featuresMapPath)
		if err != nil {
			return nil, err
		}
	}

	if featMap == nil {
		fmt.Printf("testing")
	}

	if numClasses <= 0 {
		return nil, fmt.Errorf("num class cannot be 0 or smaller: %d", numClasses)
	}

	e := &xgbEnsemble{name: "xgboost", numClasses: numClasses}
	nTrees := len(xgbEnsembleJSON)
	if nTrees == 0 {
		return nil, fmt.Errorf("no trees in file")
	} else if nTrees % e.numClasses != 0 {
		return nil, fmt.Errorf("wrong number of trees (%d) for number of class (%d)", nTrees, e.numClasses)
	}

	e.Trees = make([]*xgbTree, 0, nTrees)
	for i := 0; i < nTrees; i++ {
		tree, err := buildTree(xgbEnsembleJSON[i])
		if err != nil {
			return nil, fmt.Errorf("error while reading %d tree: %s", i, err.Error())
		}
		e.Trees = append(e.Trees, tree)
	}

	// TODO: Change transformation function.
	return &EnsembleBase{Transform: &transformation.TransformRaw{NumOutputGroups: e.numClasses}}, nil
}
