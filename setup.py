from setuptools import setup, find_packages

setup(
    name='your_project',
    version='0.1',
    packages=find_packages(where='utils'),
    package_dir={'': 'utils'},
    install_requires=[
        'pandas',
        'numpy',
        'pymatgen',
    ],
    scripts=[
        'utils/actual_usage/check_jobdir',
        'utils/actual_usage/memory_check',
        'utils/actual_usage/slurm_list_jobdir',
        'utils/actual_usage/build_and_show_db',
        'utils/actual_usage/compress_here',
        'utils/actual_usage/qstat_slurm',
        'utils/actual_usage/summarise_db',
        'utils/actual_usage/setonix_refresh_mamba',
        'utils/actual_usage/setonix_refresh_mamba',
        'utils/actual_usage/update_failed_jobs_db',
    ],
)
