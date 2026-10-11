#!/usr/bin/env bash
# Usage: bash run_all.sh FOLDER_WITH_CSVS
# Runs the three steps in order. Results go in the results/ folder.
set -e
DATA="${1:?Give the folder that holds the Telegram CSV files}"
python src/01_themes.py  "$DATA" results 12
python src/02_reposts.py "$DATA" results 0.95 168
python src/03_network.py results
