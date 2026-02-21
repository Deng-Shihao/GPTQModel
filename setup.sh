#!bin/bash

uv venv -p 3.12 && source .venv/bin/activate

uv pip install setuptools

uv pip install -v . --no-build-isolation