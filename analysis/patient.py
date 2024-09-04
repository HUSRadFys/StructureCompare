import pydicom
import glob
import seg_metrics.seg_metrics as sg
from scipy import ndimage, spatial
from math import sqrt
import numpy as np
from tqdm import tqdm
from pprint import pprint
from medpy.metric import binary

from analysis.structures import Structures

speedup = False

def find_uids_of_common_structure_length(structure_object, ds):
	uids = dict()
	max_z = -10000
	all_uids = dict()

	for rs_file in structure_object.rs_files:
		uids = dict()
		rs = structure_object.rsDict[rs_file]
		roi_idx = structure_object.ROIIdxDict[rs_file]["spinalcord"]

		for cs in rs.ROIContourSequence[roi_idx].ContourSequence:
			uid = cs.ContourImageSequence[0].ReferencedSOPInstanceUID
			z = structure_object.z[uid]

			if not uid in all_uids.values():
				all_uids[z] = uid

			uids[z] = uid

		min_z_rs_file = min(uids.keys())
		max_z = max(max_z, min_z_rs_file)

	allowed_uids = [ uid for z, uid in all_uids.items() if z >= max_z ]

	print("Number of UIDs: ", len(all_uids))
	print("Number of UIDs in caudal union: ", len(allowed_uids))

	return allowed_uids, max_z

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
		self.max_z = None

	def find_structures(self) -> None:
		self.structures_groundtruth = Structures(self.folder, self.groundtruth_folder)
		self.structures_compare = Structures(self.folder, self.compare_folder)

		common_structures = set(self.structures_groundtruth.get_structures())

		for rs_file in tqdm(self.structures_compare.rs_files):
			common_structures &= set(self.structures_compare.get_structures(rs_file))

		pprint(list(common_structures))

		self._common_structures = list(common_structures)

	def load_image_metadata(self):
		fn = glob.glob(f"{self.folder}/CT*.dcm")[0]
		self.ds = pydicom.dcmread(fn, stop_before_pixels=True)

	def get_masks(self, structure: str) -> dict:
		mask_groundtruth = self.structures_groundtruth.loadStructureMask3D(structure, self.ds)
		mask_compare = dict()
		allowed_uids = list()
		max_z = None

		if structure == "spinalcord":
			allowed_uids, max_z = find_uids_of_common_structure_length(self.structures_compare, self.ds)
			self.max_z = max_z
			mask_groundtruth = self.structures_groundtruth.loadStructureMask3D(structure, self.ds, allowed_uids=allowed_uids, max_z = max_z)

		for rs_file in self.structures_compare.rs_files:
			mask_compare[rs_file] = self.structures_compare.loadStructureMask3D(structure, self.ds, rs_file, allowed_uids, max_z = max_z)
			assert np.shape(mask_groundtruth) == np.shape(mask_compare[rs_file])

		return {'groundtruth': mask_groundtruth, 'compare': mask_compare}

	def build_metrics(self, metrics=None) -> dict:
		if not metrics:
			metrics = dict()

		metrics[self.folder] = dict()

		for structure in self.common_structures:
			if speedup and structure != "spinalcord":
				continue

			print(f"Building metrics for {self.folder} / {structure}")
			metrics[self.folder][structure] = self.build_metrics_for_structure(structure)

		return metrics

	def build_metrics_for_structure(self, structure: str) -> dict:
		masks = self.get_masks(structure)

		spacing = list(self.ds.PixelSpacing) + [self.ds.SliceThickness]
		voxel_volume = spacing[0] * spacing[1] * spacing[2] / (10**3) # cc

		metrics = dict()
		gdth = masks['groundtruth']
		com_gdth = ndimage.center_of_mass(gdth)
		mask_vol_gdth = np.sum(gdth) * voxel_volume

		for rs_file in tqdm(self.structures_compare.rs_files, disable=False):
			pred = masks['compare'][rs_file]
			metrics[rs_file] = dict()

			metrics[rs_file]["dc"] = speedup and 1 or binary.dc(gdth, pred)
			metrics[rs_file]["jc"] = speedup and 1 or binary.jc(gdth, pred)
			metrics[rs_file]["hd"] = speedup and 1 or binary.hd(gdth, pred, voxelspacing=spacing)
			metrics[rs_file]["assd"] = speedup and 1 or binary.assd(gdth, pred, voxelspacing=spacing)
			
			com_pred = ndimage.center_of_mass(pred)
			diff_3d = [k1-k2 for k1, k2 in zip(com_pred, com_gdth)]

			euclid = sqrt ( np.sum( [ ( diff_3d[i] * spacing[i] ) ** 2 for i in range(3) ] ) )

			metrics[rs_file]['center_of_mass_xyz'] = com_pred
			metrics[rs_file]['center_of_mass_difference'] = euclid
			
			mask_vol_pred = np.sum(pred) * voxel_volume

			metrics[rs_file]['volume_absolute'] = mask_vol_pred
			metrics[rs_file]['volume_difference'] = mask_vol_pred - mask_vol_gdth
			metrics[rs_file]["max_z"] = self.max_z and self.max_z or np.nan

		for rs_file in self.structures_groundtruth.rs_files:
			metrics[rs_file] = {
				'center_of_mass_xyz': com_gdth,
				'volume_absolute': mask_vol_gdth
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
	