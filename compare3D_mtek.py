from analysis.patient import Patient
from analysis.export import to_excel
from datetime import datetime
import pathlib
import glob

import cProfile, pstats, io
from pstats import SortKey

folders = list()

idx_from = 1
idx_to = 20

folder_str = "//vir-app5338.ihelse.net/va_data$/Prosjekt/MTEK_2025_AIsegmentering/PasientData/*/*"
folders = glob.glob(folder_str)

# RS to compare with needs GroundTruth in filename

metrics = None

this_idx = idx_from
for folder in folders:
	try:
		print(f"Looking at {folder = }")
		patient = Patient(folder)
		metrics = patient.build_metrics(metrics)

		dt = datetime.now().date().isoformat()
		to_excel(metrics, f'{dt}_mtek_{idx_from}_to_{this_idx}.xlsx')
	except Exception as e:
		print(f"Error: {e}; continuing")
	this_idx += 1
