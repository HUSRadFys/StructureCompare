import pandas as pd
import numpy as np
from pprint import pprint


def to_excel(metrics: dict, filename: str) -> None:
	"""Write output to Long format. """

	df = pd.DataFrame()

	for patient_raw, metrics_structure_dict in metrics.items():
		for structure, metrics_dict in metrics_structure_dict.items():
			for doseplan_name_raw, metrics in metrics_dict.items():
				doseplan_name = doseplan_name_raw.split(".")[-2].replace(" ", "")
				is_adjusted = "justering" in doseplan_name_raw.lower()

				try:
					patient = doseplan_name_raw.split(".")[-3]
				except:
					patient = patient_raw
			
				df_newrow = pd.DataFrame([{
					'Patient name': patient,
					'Is adjusted': is_adjusted,
					'Doseplan name': doseplan_name,
					'Structure name': structure,
					'Absolute volume (3D mask)': metrics['volume_absolute'],
					'Difference volume (3D mask)': metrics.get('volume_difference', np.nan),
					'Center of mass (x) [pixel]': metrics['center_of_mass_xyz'][0],
					'Center of mass (y) [pixel]': metrics['center_of_mass_xyz'][1],
					'Center of mass (z) [pixel]': metrics['center_of_mass_xyz'][2],
					'Center of mass (3D difference) [mm]': metrics.get('center_of_mass_difference', np.nan),
					'DICE coefficient (3D)': metrics.get('dc', np.nan),
					'Hausdorff distance (2D) [mm]': metrics.get('hd', np.nan),
					'Hausdorff 95 percentile (2D) [mm]': metrics.get('hd95', np.nan),
					'Jaccard index (3D)': metrics.get("jc", np.nan),
					'Average Symmetric Surface Distance (2D) [mm]': metrics.get("assd", np.nan),
					"Max Z spinalcord cutoff [mm]": metrics.get("max_z", np.nan),
				}])

				df = pd.concat([df, df_newrow])

	pprint(df)
	df.to_excel(filename, index=False)	