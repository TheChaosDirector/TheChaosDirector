#!/bin/bash
# Pack the current portfolio champion into a small zip for CI / GitHub Releases.
# Usage:
#   ./scripts/pack-champion.sh
#   ./scripts/pack-champion.sh --publish   # needs: gh auth + repo write access
set -euo pipefail
cd "$(dirname "$0")/.."

META="artifacts/portfolio/registry/champion.json"
if [[ ! -f "$META" ]]; then
  META="artifacts/registry/champion.json"
fi
if [[ ! -f "$META" ]]; then
  echo "No champion found. Train first: python -m src.cli train --config configs/default.yaml --mode portfolio"
  exit 1
fi

MODEL_PATH="$(python3 -c "import json; print(json.load(open('$META'))['model_path'])")"
if [[ ! -f "$MODEL_PATH" ]]; then
  # Prefer mode-scoped copy when legacy path is missing.
  FOLD="$(python3 -c "import json; print(json.load(open('$META'))['fold'])")"
  ALT="artifacts/portfolio/registry/fold_$(printf '%02d' "$FOLD")/model.zip"
  if [[ -f "$ALT" ]]; then
    MODEL_PATH="$ALT"
  else
    echo "Champion model file missing: $MODEL_PATH"
    exit 1
  fi
fi

OUT_DIR="artifacts/bundles"
mkdir -p "$OUT_DIR"
ZIP="$OUT_DIR/portfolio-champion.zip"
rm -f "$ZIP"

# Normalize paths inside the zip so CI always unpacks to artifacts/portfolio/registry/
STAGE="$(mktemp -d)"
FOLD_DIR="$(dirname "$MODEL_PATH")"
FOLD_NAME="$(basename "$FOLD_DIR")"
mkdir -p "$STAGE/artifacts/portfolio/registry/$FOLD_NAME"
cp "$MODEL_PATH" "$STAGE/artifacts/portfolio/registry/$FOLD_NAME/model.zip"
python3 - <<PY
import json
from pathlib import Path
meta = json.loads(Path("$META").read_text())
fold = int(meta["fold"])
meta["model_path"] = f"artifacts/portfolio/registry/fold_{fold:02d}/model.zip"
Path("$STAGE/artifacts/portfolio/registry/champion.json").write_text(json.dumps(meta, indent=2, default=str))
PY

( cd "$STAGE" && zip -qr "$OLDPWD/$ZIP" artifacts )
rm -rf "$STAGE"
echo "Wrote $ZIP ($(du -h "$ZIP" | cut -f1))"

if [[ "${1:-}" == "--publish" ]]; then
  TAG="portfolio-champion"
  TITLE="Portfolio champion bundle"
  # Recreate the lightweight release tag used by the daily GitHub Action.
  if gh release view "$TAG" >/dev/null 2>&1; then
    gh release delete "$TAG" --yes
    git push origin ":refs/tags/$TAG" 2>/dev/null || true
  fi
  gh release create "$TAG" "$ZIP" --title "$TITLE" --notes "Frozen portfolio champion for daily Alpaca PAPER automation."
  echo "Published GitHub release tag: $TAG"
fi
