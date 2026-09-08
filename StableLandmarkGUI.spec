# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['stable_landmark_project/stable_landmark_gui.py'],
    pathex=['stable_landmark_project'],
    binaries=[],
    datas=[('stable_landmark_project/patient461_known_landmarks.json', '.')],
    hiddenimports=['stable_landmark_labeler_v2'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='StableLandmarkGUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='StableLandmarkGUI',
)
app = BUNDLE(
    coll,
    name='StableLandmarkGUI.app',
    icon=None,
    bundle_identifier=None,
)
