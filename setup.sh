#!/bin/bash

set -e

mkdir -p ./res
mkdir -p ./code/{src,tests,docs}

touch ./code/src/__init__.py
touch ./code/tests/__init__.py
touch ./code/src/main.py
touch ./code/tests/stub.py

touch ./code/docs/README.md
touch ./logs.md