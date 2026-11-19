# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from importlib.metadata import version

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

APP_NAME = "Screen Translator"
EXE_NAME = "screen-translator"
BUNDLE_ID = "io.github.khoale38.screentranslator"
VERSION = version("screen-translator")

datas = collect_data_files("screen_translator") + collect_data_files("llama_cpp")
binaries = collect_dynamic_libs("llama_cpp")
hiddenimports = collect_submodules("screen_translator") + collect_submodules(
    "llama_cpp", filter=lambda name: not name.startswith("llama_cpp.server")
)

if sys.platform == "darwin":
    hiddenimports += ["objc", "Vision", "Quartz"]
else:
    datas += collect_data_files("rapidocr_onnxruntime")
    binaries += collect_dynamic_libs("onnxruntime")
    hiddenimports += collect_submodules("rapidocr_onnxruntime")

a = Analysis(
    [os.path.join(SPECPATH, "launcher.py")],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "pytest", "IPython", "matplotlib", "PyQt5", "PyQt6", "PySide2", "PySide6", "gi"],
    noarchive=False,
)
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
)

coll = COLLECT(exe, a.binaries, a.datas, name=EXE_NAME, upx=False)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        bundle_identifier=BUNDLE_ID,
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
