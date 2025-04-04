from utils.vasp.vasp_resubmitter import CalculationConverger
import os

vasp_resubmitter = CalculationConverger(
    parent_dir=os.getcwd(),
    script_template_dir="/home/hmai/CustodianJobfiles",
    max_submissions=1000,
    submission_command="sbatch",
    username="hmai",
)

vasp_resubmitter.reconverge_all()
