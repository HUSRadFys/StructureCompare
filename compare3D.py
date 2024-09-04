from analysis.patient import Patient
from analysis.export import to_excel

#folders = glob.glob("data/*")

folders = ["data/zzART_Head1"]
folders = list()

base = "//vir-app5338.ihelse.net/va_data$/Prosjekt/Aicontouring/"
for k in range(1, 7): # Trouble with pat 7 for now,
	folders += base + f"zzART_Head{k}_Forskning/zzART_Head{k}",

metrics = None

for folder in folders:
	patient = Patient(folder)
	metrics = patient.build_metrics(metrics)

to_excel(metrics, '2024-09-04_Patient_1_to_6_spinalcord_matching.xlsx')