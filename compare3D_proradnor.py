from analysis.patient import Patient
from analysis.export import to_excel
from datetime import datetime

import cProfile, pstats, io
from pstats import SortKey

folders = list()

idx_from = 1
idx_to = 21

folder = "//vir-app5338.ihelse.net/va_data$/Prosjekt/Proradnor_Inntegning_analyse/zzPRORADNOR_Tromsø1/"

# RS to compare with needs GroundTruth in filename

metrics = None

try:
	patient = Patient(folder)
	metrics = patient.build_metrics(metrics)

	dt = datetime.now().date().isoformat()
	to_excel(metrics, f'{dt}_proradnor.xlsx')
except Exception as e:
	print(f"Error: {e}; continuing")