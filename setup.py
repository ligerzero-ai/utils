from setuptools import setup, find_packages

setup(
    name="utils",
    version="0.1",
    packages=find_packages(where="utils"),
    package_dir={"": "utils"},
    install_requires=[
        "pandas",
        "numpy",
        "pymatgen",
    ],
    scripts=[
        "actual_usage/check_jobdir",
        "actual_usage/memory_check",
        "actual_usage/slurm_list_jobdir",
        "actual_usage/build_and_show_db",
        "actual_usage/compress_here",
        "actual_usage/qstat_slurm",
        "actual_usage/summarise_db",
        "actual_usage/setonix_refresh_mamba",
        "actual_usage/setonix_refresh_mamba",
        "actual_usage/update_failed_jobs_db",
    ],
)
