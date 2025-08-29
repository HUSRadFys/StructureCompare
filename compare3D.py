from analysis.patient import Patient
from analysis.export import to_excel
from datetime import datetime

import cProfile, pstats, io
from pstats import SortKey

folders = list()

idx_from = 1
idx_to = 21

base = "//vir-app5338.ihelse.net/va_data$/Prosjekt/Aicontouring/"
for k in range(idx_from,idx_to): # Trouble with pat 7 for now,
	folders += base + f"zzART_Head{k}_Forskning/zzART_Head{k}",

metrics = None

"""
pr = cProfile.Profile()
pr.enable()
"""

# RS to compare with needs GroundTruth in filename


this_idx = idx_from
for folder in folders:
	try:
		patient = Patient(folder)
		metrics = patient.build_metrics(metrics)

		dt = datetime.now().date().isoformat()
		to_excel(metrics, f'{dt}_hd95_Patient_{idx_from}_to_{this_idx}.xlsx')
	except Exception as e:
		print(f"Error: {e}; continuing")
	this_idx += 1


"""
pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
"""


# to_excel(metrics, f'{dt}_Patient_18.xlsx')