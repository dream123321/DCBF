# setup.py
from pathlib import Path
import os

from setuptools import Extension, setup, find_packages


selection_dir = Path(__file__).resolve().parent / "dcbf" / "selection"
prebuilt_selection_extensions = []
for pattern in ("_min_cover_exact*.so", "_min_cover_exact*.pyd", "_min_cover_exact*.dylib"):
    prebuilt_selection_extensions.extend(path.name for path in selection_dir.glob(pattern))
prebuilt_selection_extensions = sorted(set(prebuilt_selection_extensions))

selection_extension = Extension(
    "dcbf.selection._min_cover_exact",
    sources=["dcbf/selection/_min_cover_exact.cpp"],
    language="c++",
    extra_compile_args=["-O3"],
)

use_prebuilt_selection_extension = bool(prebuilt_selection_extensions) and os.environ.get("DCBF_FORCE_BUILD_EXT") != "1"
ext_modules = [] if use_prebuilt_selection_extension else [selection_extension]

setup(
    name='dcbf',
    version='1.0',
    packages=find_packages(include=['dcbf', 'dcbf.*']),
    package_data={
        'dcbf': [
            'mtp_templates/*.mtp',
            'default_reduce_assets/*.mtp',
            'default_reduce_assets/*.txt',
            'default_reduce_assets/*.pth',
            'training_assets/*.mtp',
            'training_assets/*.py',
        ],
        'dcbf.selection': prebuilt_selection_extensions,
    },
    author='Jing Huang',
    author_email='2760344463@qq.com',
    description='DCBF active-learning workflow',
    install_requires=[
    ],
    ext_modules=ext_modules,
    entry_points={
        'console_scripts': [
            'dcbf=dcbf.cli:main',
            'dcbf-predict-xyz=dcbf.high_precision_tools:predict_xyz_main',
            'dcbf-plot-errors=dcbf.high_precision_tools:plot_errors_main',
        ]
    }
)


'''
Remark:
install pymlip
install vaspkit
'ase==3.23.0',
'''
