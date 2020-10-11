#!/bin/bash

echo "${GOPATH}:."

protoc \
    --proto_path=${GOPATH}/src/:${GOPATH}/src/github.com/gogo/protobuf/protobuf/:. \
    --gofast_out=. *.proto