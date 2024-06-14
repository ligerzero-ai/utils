from setuptools import setup

setup(
    name='your_project',
    version='0.1',
    packages=[
        'vasp',
        'plotters',
        'actual_usage',
        'WIP_notebooks',
        'GNN_calculators',
        'jobscript_templates',
        'StructureManipulator'
    ],
    install_requires=[
        'pandas',
        'numpy',
        'pymatgen',
    ],
)
