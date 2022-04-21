#!/bin/bash

PARENT_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

cd "$PARENT_PATH"

CSAR_URL="1u0EpAi7d9rF7j8RdkMCFROSXh8RNhXCo"

gdown --id $CSAR_URL --output "../data/csar_processed.zip"
