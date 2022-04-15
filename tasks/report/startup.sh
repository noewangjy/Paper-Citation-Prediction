#!/bin/bash

LINK=https://github.com/davidliyutong/ICE6407P-260-M01/releases/download/submission/features.zip
FLATTERN_DIR=./features/

set -e
wget $LINK -O features.zip
unzip features.zip
