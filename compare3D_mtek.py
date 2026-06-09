from analysis.patient import Patient
from analysis.export import to_excel
from datetime import datetime
import pathlib
import glob
from pprint import pprint

import cProfile, pstats, io
from pstats import SortKey

folders = list()

#folder_str = "//vir-app5338.ihelse.net/va_data$/Prosjekt/MTEK_2025_AIsegmentering/PasientData_2025/*/*"
folder_str = "//vir-app5338.ihelse.net/va_data$/Prosjekt/SyngoTest/*"
folders = glob.glob(folder_str)

pprint(folders)

# RS to compare with needs GroundTruth in filename

metrics = None

for folder in folders:
	try:
		print(f"Looking at {folder = }")
		patient = Patient(folder)
		metrics = patient.build_metrics(metrics)

		dt = datetime.now().date().isoformat()
		to_excel(metrics, f'Output/{dt}_mtek_2025_fix.xlsx')
	except Exception as e:
		print(f"Error: {e}; continuing")