import pandas as pd
import numpy as np

from pprint import pprint


def to_excel(metrics: dict, filename: str) -> None:
	"""Write output to Long format. """

	df = pd.DataFrame()

	for structure, metrics_dict in metrics.items():
		for doseplan_name_raw, metrics in metrics_dict.items():
			doseplan_name = doseplan_name_raw.split(".")[-2].replace(" ", "")
			patient = doseplan_name_raw.split(".")[-3]

			if 'sg' in metrics:
				sg = metrics['sg'][0]
			else:
				sg = None
		
			df_newrow = pd.DataFrame([{
				'Patient name': patient,
				'Doseplan name': doseplan_name,
				'Structure name': structure,
				'Absolute volume (3D interpolated)': metrics['dvh_volume_absolute'],
				'Difference volume (3D interpolated)': metrics.get('dvh_volume_difference', np.nan),
				'Absolute volume (3D mask)': metrics['mask_volume_absolute'],
				'Difference volume (3D mask)': metrics.get('mask_volume_difference', np.nan),
				'Center of mass (x) [pixel]': metrics['center_of_mass_xyz'][0],
				'Center of mass (y) [pixel]': metrics['center_of_mass_xyz'][1],
				'Center of mass (z) [pixel]': metrics['center_of_mass_xyz'][2],
				'Center of mass (3D difference) [mm]': metrics.get('center_of_mass', np.nan),
				'DICE coefficient (3D)': sg and sg['dice'][0] or np.nan,
				'Hausdorff distance (2D) [mm]': sg and sg['hd'][0] or np.nan,
				'Hausdorff 95 percentile (2D) [mm]': sg and sg['hd95'][0] or np.nan,
				'Jaccard index (3D)': sg and sg['jaccard'][0] or np.nan,
				'Mean Surface Distance (2D) [mm]': sg and sg['msd'][0] or np.nan
			}])

			df = pd.concat([df, df_newrow])

	pprint(df)
	df.to_excel(filename)
