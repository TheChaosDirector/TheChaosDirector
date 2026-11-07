#!/bin/bash
# Save a local copy of the published live champion.
# Does not train, does not publish, and does not write into artifacts/portfolio.
set -euo pipefail
cd "$(dirname "$0")/.."

DEST="artifacts/original-champion"
mkdir -p "$DEST"
gh release download portfolio-champion \
  --pattern 'portfolio-champion.zip' \
  --dir "$DEST" \
  --clobber
unzip -o "$DEST/portfolio-champion.zip" -d "$DEST"
test -f "$DEST/artifacts/portfolio/registry/champion.json"
echo "Live champion snapshot: $DEST"
echo "The daily paper job still downloads the portfolio-champion release."
echo "The Treasury copy trains under artifacts/yield-compete/."
