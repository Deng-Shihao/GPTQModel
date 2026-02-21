#!bin/bash

uv venv -p 3.12 && source .venv/bin/activate

uv pip install setuptools torch

uv pip install --upgrade pip setuptools wheel

uv pip install -v . --no-build-isolation