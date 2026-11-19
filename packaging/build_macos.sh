#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-.venv/bin/python}
APP="dist/Screen Translator.app"
VERSION=$("$PYTHON" -c 'from importlib.metadata import version; print(version("screen-translator"))')
DMG="dist/ScreenTranslator-${VERSION}-macos-arm64.dmg"

"$PYTHON" -m PyInstaller --noconfirm --clean --distpath dist --workpath build/pyinstaller packaging/screen_translator.spec
codesign --force --deep --sign - "$APP"
codesign --verify --deep --strict "$APP"

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
ditto "$APP" "$STAGE/Screen Translator.app"
ln -s /Applications "$STAGE/Applications"
rm -f "$DMG"
hdiutil create -volname "Screen Translator" -srcfolder "$STAGE" -fs HFS+ -format UDZO -ov "$DMG"
echo "Built $APP and $DMG"
