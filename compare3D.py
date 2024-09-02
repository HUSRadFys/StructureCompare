from analysis.patient import Patient
from analysis.export import to_excel

#folders = glob.glob("data/*")

folders = ["data/zzART_Head1"]

base = "//vir-app5338.ihelse.net/va_data$/Prosjekt/Aicontouring/"
for k in range(2, 5):
	folders += base + f"zzART_Head{k}_Forskning/zzART_Head{k}",

metrics = None
for folder in folders:
	patient = Patient(folder)
	metrics = patient.build_metrics(metrics)

to_excel(metrics, 'zzART_Head1-4_withpharynxconstrict.xlsx')