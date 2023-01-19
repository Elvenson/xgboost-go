# xgboost-go

[![Build Status](https://travis-ci.com/Elvenson/xgboost-go.svg?token=rzHXU1xSU77dfjTLof6x&branch=main)](https://travis-ci.com/github/Elvenson/xgboost-go)
[![GoDoc](https://godoc.org/github.com/Elvenson/xgboost-go?status.png)](https://godoc.org/github.com/Elvenson/xgboost-go)

XGBoost inference with Golang by means of exporting xgboost model into json format and load model from that json file. 
This repo only supports [DMLC XGBoost](https://github.com/dmlc/xgboost) model at the moment. For more information regarding 
how XGBoost inference works, you can refer to this [medium article](https://medium.com/@bobi_29852/how-boosted-trees-inference-works-f161b03d5f5b).

## Features
Currently, this repo only supports a few core features such as:

* Read models from json format file (via `dump_model` API call)
* Support sigmoid and softmax transformation activation.
* Support binary and multiclass predictions.
* Support regressions predictions.
* Support missing values.
* Support libsvm data format.

**NOTE**: The result from DMLC XGBoost model may slightly differ from this model due to float number precision.

## How to use:
To use this repo, first you need to get it:
```shell script
go get github.com/Elvenson/xgboost-go
```

Basic example:

```go
package main

import (
	"fmt"

	xgb "github.com/Elvenson/xgboost-go"
	"github.com/Elvenson/xgboost-go/activation"
	"github.com/Elvenson/xgboost-go/mat"
)

func main() {
	ensemble, err := xgb.LoadXGBoostFromJSON("your model path",
		"", 1, 4, &activation.Logistic{})
	if err != nil {
		panic(err)
	}

	input, err := mat.ReadLibsvmFileToSparseMatrix("your libsvm input path")
	if err != nil {
		panic(err)
	}
	predictions, err := ensemble.PredictProba(input)
	if err != nil {
		panic(err)
	}
	fmt.Printf("%+v\n", predictions)
}
```

Here `LoadXGBoostFromJSON` requires 5 parameters:
* The json model path.
* DMLC feature map format, if no feature map leave this blank.
* The number of classes (if this is a binary classification, the number of classes should be 1)
* The depth of the tree, if unable to get the tree depth can specify 0 (slightly slower model built time)
* Activation function, for now binary is `Logistic` multiclass is `Softmax` and regression is `Raw`.

For more example, can take a look at `xgbensemble_test.go` or read this package
[documentation](https://godoc.org/github.com/Elvenson/xgboost-go).

**NOTE**: This repo only got tested on Python `xgboost` package version `1.2.0`.
