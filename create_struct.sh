#!/bin/bash

# Create main directories and subdirectories
mkdir -p image-to-number/{data/{raw,training},docs,src,experiments/{logs,checkpoints},tests,scripts}

# Create placeholder files
touch image-to-number/README.md
touch image-to-number/LICENSE
touch image-to-number/requirements.txt
touch image-to-number/setup.py

touch image-to-number/docs/design.md
touch image-to-number/docs/model_architecture.md

touch image-to-number/src/__init__.py
touch image-to-number/src/cli.py
touch image-to-number/src/model.py
touch image-to-number/src/train.py
touch image-to-number/src/inference.py
touch image-to-number/src/utils.py

touch image-to-number/tests/__init__.py
touch image-to-number/tests/test_model.py
touch image-to-number/tests/test_inference.py
touch image-to-number/tests/test_cli.py

touch image-to-number/experiments/config.yaml

touch image-to-number/scripts/run.sh
touch image-to-number/scripts/train.sh

echo "Project structure created successfully."
