import pydicom
import glob
import seg_metrics.seg_metrics as sg
from scipy import ndimage
from math import sqrt
import numpy as np
from tqdm import tqdm

from analysis.structures import Structures

class Patient:
	def __init__(self, folder: str) -> None:
		self.folder = folder
		self.compare_folder = 'RS_compare'
		self.groundtruth_folder = 'RS_groundtruth'
		self.metrics_to_use = [
			'dice',
			'jaccard',
			'hd',
			'hd95',
			'msd'
		]

		self.volume_groundtruth = dict()

		self.find_structures()
		self.load_image_metadata()

	def find_structures(self) -> None:
		self.structures_groundtruth = Structures(self.folder, self.groundtruth_folder)
		self.structures_compare = Structures(self.folder, self.compare_folder)

		# TODO: use .get_structures(rsFilename), loop over rsFilename
		common_structures = set(self.structures_groundtruth.get_structures())

		# print(f"pharynx in Ground Truth: ", end="")
		# print([k for k in self.structures_groundtruth.get_structures() if "pharynx" in k])

		print("Finding common structures")

		for rs_file in tqdm(self.structures_compare.rs_files):
			common_structures &= set(self.structures_compare.get_structures(rs_file))
			# print(f"pharynx in {rs_file}: ", end="")
			# print([k for k in self.structures_compare.get_structures(rs_file) if "pharynx" in k])

		self._common_structures = list(common_structures)

	def load_image_metadata(self):
		fn = glob.glob(f"{self.folder}/CT*.dcm")[0]
		self.ds = pydicom.dcmread(fn, stop_before_pixels=True)

	def get_masks(self, structure: str) -> dict:
		mask_groundtruth = self.structures_groundtruth.loadStructureMask3D(structure, self.ds)
		mask_compare = dict()
		for rs_file in self.structures_compare.rs_files:
			mask_compare[rs_file] = self.structures_compare.loadStructureMask3D(structure, self.ds, rs_file)

		return {'groundtruth': mask_groundtruth, 'compare': mask_compare}

	def build_metrics(self, metrics=None) -> dict:
		if not metrics:
			metrics = dict()

		metrics[self.folder] = dict()

		for structure in self.common_structures:
			print(f"Building metrics for {self.folder} / {structure}")
			metrics[self.folder][structure] = self.build_metrics_for_structure(structure)

		return metrics

	def build_metrics_for_structure(self, structure: str) -> dict:
		masks = self.get_masks(structure)

		spacing = list(self.ds.PixelSpacing) + [self.ds.SliceThickness]
		voxel_volume = spacing[0] * spacing[1] * spacing[2] / (10**3) # cc

		metrics = dict()
		print("Looping through contours")
		for rs_file in tqdm(self.structures_compare.rs_files):
			metrics[rs_file] = {'sg': sg.write_metrics(labels=[1], 
			                              gdth_img=masks['groundtruth'],
			                              pred_img=masks['compare'][rs_file],
			                              csv_file=None, 
			                              spacing=spacing, 
			                              metrics=self.metrics_to_use, 
			                              verbose=False
			) }

			cm = self.get_center_of_mass_difference(masks, rs_file)
			metrics[rs_file]['center_of_mass_xyz'] = cm[0]
			metrics[rs_file]['center_of_mass'] = cm[1]

			volumes = self.get_volumes(structure, rs_file)
			metrics[rs_file]['dvh_volume_absolute'] = volumes['absolute']
			metrics[rs_file]['dvh_volume_difference'] = volumes['difference']

			mask_vol_compare = np.sum(masks['compare'][rs_file]) * voxel_volume
			mask_vol_groundtruth = np.sum(masks['groundtruth']) * voxel_volume

			metrics[rs_file]['mask_volume_absolute'] = mask_vol_compare
			metrics[rs_file]['mask_volume_difference'] = mask_vol_compare - mask_vol_groundtruth

		for rs_file in self.structures_groundtruth.rs_files:
			metrics[rs_file] = {
				'center_of_mass_xyz': self.get_center_of_mass_difference(masks, rs_file),
				'dvh_volume_absolute': self.get_volumes(structure, rs_file),
				'mask_volume_absolute': np.sum(masks['groundtruth']) * voxel_volume
			}

		return metrics
	
	def get_center_of_mass_difference(self, mask: dict, rs_file: str) -> float:
		cm_groundtruth = ndimage.center_of_mass(mask['groundtruth'])

		if not rs_file in self.structures_compare.rs_files:
			return cm_groundtruth

		cm_compare = ndimage.center_of_mass(mask['compare'][rs_file])

		diff = [k1-k2 for k1,k2 in zip(cm_groundtruth, cm_compare)]
		euclid_diff = sqrt((diff[0] * self.ds.PixelSpacing[0])**2 + 
								 (diff[1] * self.ds.PixelSpacing[1])**2 +
								 (diff[2] * self.ds.SliceThickness)**2)
		
		return cm_compare, euclid_diff

	def get_volumes(self, structure: str, rs_file: str) -> dict:
		""" Calculate volume with DVH methods."""
		if not self.volume_groundtruth.get(structure):
			self.volume_groundtruth[structure] = self.structures_groundtruth.get_volume(structure)

		if not rs_file in self.structures_compare.rs_files:
			return self.volume_groundtruth.get(structure)

		volume = self.structures_compare.get_volume(structure, rs_file)

		return {'absolute': volume, 'difference': volume - self.volume_groundtruth.get(structure)}

	@property
	def common_structures(self) -> list:
		return self._common_structures
	