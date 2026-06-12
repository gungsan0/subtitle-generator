#!/bin/bash
# Build the Subtitle Generator .pkg installer
set -e
cd "$(dirname "$0")"

VERSION="${1:-1.0.0}"
APP_ROOT="root/usr/local/subtitle-generator"

# Copy current app files into the payload
rm -rf "$APP_ROOT"
mkdir -p "$APP_ROOT/static"
cp ../main.py "$APP_ROOT/"
cp ../requirements.txt "$APP_ROOT/"
cp ../static/index.html "$APP_ROOT/static/"

chmod +x scripts/postinstall root/usr/local/bin/subtitle-generator

# Strip macOS metadata files from the payload.
# Note: com.apple.provenance xattrs may appear as ._ entries in the payload
# listing — these are converted back to xattrs at install time and are harmless.
find root -name '._*' -delete
find root -name '.DS_Store' -delete

pkgbuild \
  --root root \
  --scripts scripts \
  --identifier com.gungsan0.subtitle-generator \
  --version "$VERSION" \
  --install-location / \
  "SubtitleGenerator-$VERSION.pkg"

echo ""
echo "Built: packaging/SubtitleGenerator-$VERSION.pkg"
