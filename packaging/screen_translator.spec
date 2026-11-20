# -*- mode: python ; coding: utf-8 -*-
import glob
import os
import subprocess
import sys
from importlib.metadata import version

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

APP_NAME = "Screen Translator"
EXE_NAME = "screen-translator"
BUNDLE_ID = "io.github.khoale38.screentranslator"
VERSION = version("screen-translator")
ICONS = os.path.join(SPECPATH, "icons")

datas = collect_data_files("screen_translator") + collect_data_files("llama_cpp")
binaries = collect_dynamic_libs("llama_cpp")
hiddenimports = collect_submodules("screen_translator") + collect_submodules(
    "llama_cpp", filter=lambda name: not name.startswith("llama_cpp.server")
)

excludes = ["tkinter", "pytest", "IPython", "matplotlib", "PyQt5", "PyQt6", "PySide2", "gi", "webview"]

if sys.platform == "darwin":
    hiddenimports += ["objc", "Vision", "Quartz", "AppKit"]
if sys.platform == "darwin" and os.environ.get("SCREEN_TRANSLATOR_OCR") != "rapidocr":
    excludes += ["rapidocr", "onnxruntime", "cv2"]
else:
    ocr_models = os.path.join(SPECPATH, os.pardir, "build", "ocr_models")
    if not glob.glob(os.path.join(ocr_models, "*.onnx")):
        subprocess.run([sys.executable, "-m", "screen_translator", "--prefetch-ocr", ocr_models], check=True)
    datas += collect_data_files("rapidocr", excludes=["models/*.onnx"]) + [(ocr_models, "ocr_models")]
    binaries += collect_dynamic_libs("onnxruntime")
    hiddenimports += collect_submodules("rapidocr")

a = Analysis(
    [os.path.join(SPECPATH, "launcher.py")],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
# The Vulkan loader ships with GPU drivers; a bundled copy would shadow the system one.
a.binaries = [entry for entry in a.binaries if os.path.basename(entry[0]).lower() != "vulkan-1.dll"]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=EXE_NAME,
    console=False,
    disable_windowed_traceback=False,
    upx=False,
    icon=os.path.join(ICONS, "icon.ico"),
)

coll = COLLECT(exe, a.binaries, a.datas, name=EXE_NAME, upx=False)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        bundle_identifier=BUNDLE_ID,
        icon=os.path.join(ICONS, "icon.icns"),
        version=VERSION,
        info_plist={
            "CFBundleName": APP_NAME,
            "CFBundleDisplayName": APP_NAME,
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "LSMinimumSystemVersion": "13.0",
            "LSApplicationCategoryType": "public.app-category.utilities",
            "NSHighResolutionCapable": True,
            "NSAppTransportSecurity": {"NSAllowsLocalNetworking": True},
            "NSScreenCaptureUsageDescription": (
                "Screen Translator reads the text in the screen area you select so it can translate it. "
                "Everything stays on this Mac."
            ),
        },
    )
