# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for BodySegAI Windows build

import os
import sys

block_cipher = None

# Paths
ROOT = os.path.abspath('.')
MODEL_PATH = os.path.join(ROOT, 'xRobotstuffx', 'model.onnx')

a = Analysis(
    ['main.py'],
    pathex=[ROOT],
    binaries=[],
    datas=[
        (MODEL_PATH, 'xRobotstuffx'),
        (os.path.join(ROOT, 'bodysegai', 'templates'), 'bodysegai/templates'),
        (os.path.join(ROOT, 'bodysegai', 'static'), 'bodysegai/static'),
        (os.path.join(ROOT, 'licence.pdf'), '.'),
    ],
    hiddenimports=[
        'onnxruntime',
        'nibabel',
        'nibabel.nifti1',
        'nibabel.nifti2',
        'nibabel.gifti',
        'nibabel.freesurfer',
        'pydicom',
        'pydicom.encoders',
        'pydicom.encoders.gdcm',
        'pydicom.encoders.pylibjpeg',
        'SimpleITK',
        'skimage',
        'skimage.transform',
        'PIL',
        'reportlab',
        'reportlab.lib',
        'reportlab.platypus',
        'reportlab.graphics',
        'flask',
        'jinja2',
        'numpy',
        'scipy',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tensorflow',
        'keras',
        'torch',
        'matplotlib',
        'tkinter',
        'IPython',
        'jupyter',
        'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BodySegAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window
    icon=None,  # TODO: add icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BodySegAI',
)
