#!/usr/bin/env bash

pre-commit run -a && mypy --install-types src tests
