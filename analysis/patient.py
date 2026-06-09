import pydicom
import glob
from scipy.stats import t
import seg_metrics.seg_metrics as sg
from scipy import ndimage, spatial
from math import sqrt
import numpy as np
from tqdm import tqdm
from pprint import pprint
from medpy.metric import binary
from time import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from analysis.structures import Structures

def find_uids_of_common_structure_length(structure_object, ds):
	uids = dict()
	max_z = -10000
	all_uids = dict()

	for rs_file in structure_object.rs_files:
		uids = dict()
		rs = structure_object.rsDict[rs_file]
		roi_idx = structure_object.ROIIdxDict[rs_file]["spinalcord"]

		if not "ContourSequence" in rs.ROIContourSequence[roi_idx]:
			return list(), 0

		for cs in rs.ROIContourSequence[roi_idx].ContourSequence:
			uid = cs.ContourImageSequence[0].ReferencedSOPInstanceUID
			# z = structure_object.z[uid]
			z = cs.ContourData[2]
			# print("z from uid:", z)
			# print("z from data:", cs.ContourData[2])

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
		try:
			self.load_image_metadata()
		except:
			print("Cannot load CT images")
			self.ds = list()
		self.max_z = None

	def find_structures(self) -> None:
		self.structures_groundtruth = Structures(self.folder, self.groundtruth_folder)
		self.structures_compare = Structures(self.folder, self.compare_folder)

		print("Referenced Frame of References (gdth)")
		pprint(self.structures_groundtruth.frame_of_reference_uid)
		print("Referenced Frame of References (pred)")
		pprint(self.structures_compare.frame_of_reference_uid)

		print("Structures in GROUND TRUTH: ")
		pprint(self.structures_groundtruth.get_structures())

		common_structures = set(self.structures_groundtruth.get_structures())

		mapping = { # -> GT structure name
			"femur_head_l": "femuralhead_l",
			"femur_head_r": "femuralhead_r",
			"femur_lprox": "femuralhead_l",
			"femur_rprox": "femuralhead_r",
			"anorectum": "rectum",
		}

		self.mapped = {k: dict() for k in self.structures_compare.rs_files}

		for rs_file in tqdm(self.structures_compare.rs_files):

			print(f"Structures in {rs_file}: ", end="")
			compare_structures = self.structures_compare.get_structures(rs_file)
			pprint(compare_structures)
			
			# Inverse mapping of ALL structures to obtain original name later
			for s in compare_structures:
				self.mapped[rs_file][mapping.get(s, s)] = s

			common_structures &= set(self.mapped[rs_file].keys())

		pprint(list(common_structures))

		self._common_structures = list(common_structures)

	def load_image_metadata(self):
		fn = glob.glob(f"{self.folder}/CT*.dcm")[0]
		self.ds = pydicom.dcmread(fn, stop_before_pixels=True)

	"""
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
	"""

	def build_metrics(self, metrics=None) -> dict:
		if not metrics:
			metrics = dict()

		metrics[self.folder] = dict()

		for structure in self.common_structures:
			if structure == "body":
				continue

			print(f"Building metrics for {self.folder} / {structure}")
			metrics[self.folder][structure] = self.build_metrics_for_structure(structure)

		return metrics

	def build_metrics_for_structure(self, structure: str) -> dict:
		# masks = self.get_masks(structure)
		allowed_uids = list()
		max_z = None
		
		if structure == "spinalcord":
			allowed_uids, max_z = find_uids_of_common_structure_length(self.structures_compare, self.ds)
			gdth = self.structures_groundtruth.loadStructureMask3D(structure, self.ds, allowed_uids=allowed_uids, max_z = max_z)
		else:
			gdth = self.structures_groundtruth.loadStructureMask3D(structure, self.ds)

		spacing = [self.ds.SliceThickness] + list(self.ds.PixelSpacing)[::-1]

		voxel_volume = spacing[0] * spacing[1] * spacing[2] / (10**3) # cc

		metrics = dict()
		# gdth = masks['groundtruth']
		com_gdth = ndimage.center_of_mass(gdth)
		mask_vol_gdth = np.sum(gdth) * voxel_volume

		print(f"Parsing {self.structures_compare.rs_files = }")

		for rs_file in tqdm(self.structures_compare.rs_files, disable=False):
			mapped_structure_name = self.mapped[rs_file][structure]
			print(f"Using mapped structure name {mapped_structure_name} for {structure} in {rs_file}")
			pred = self.structures_compare.loadStructureMask3D(mapped_structure_name, self.ds, rs_file, allowed_uids, max_z = max_z)
			metrics[rs_file] = dict()

			if not np.sum(pred):
				print("Empty structure file")
				metrics[rs_file]["dc"] = metrics[rs_file]["jc"] = "EMPTY STRUCTURE FILE"

			metrics[rs_file]["dc"] = binary.dc(gdth, pred)
			metrics[rs_file]["jc"] = binary.jc(gdth, pred)
			metrics[rs_file]["hd"] = binary.hd(gdth, pred, voxelspacing=spacing)
			metrics[rs_file]["hd95"] = binary.hd95(gdth, pred, voxelspacing=spacing)
			metrics[rs_file]["assd"] = binary.assd(gdth, pred, voxelspacing=spacing)
			
			com_pred = ndimage.center_of_mass(pred)
			diff_3d = [k1-k2 for k1, k2 in zip(com_pred, com_gdth)]

			euclid = sqrt ( np.sum( [ ( diff_3d[i] * spacing[i] ) ** 2 for i in range(3) ] ) )
			metrics[rs_file]['center_of_mass_xyz'] = com_pred
			
			metrics[rs_file]['center_of_mass_difference'] = euclid
			
			metrics[rs_file]['center_of_mass_xyz'] = [0,0,0]

			mask_vol_pred = np.sum(pred) * voxel_volume

			metrics[rs_file]['volume_absolute'] = mask_vol_pred
			metrics[rs_file]['volume_difference'] = mask_vol_pred - mask_vol_gdth
			metrics[rs_file]["max_z"] = max_z and max_z or np.nan			

		for rs_file in self.structures_groundtruth.rs_files:
			metrics[rs_file] = {
				'center_of_mass_xyz': com_gdth,
				# 'center_of_mass_xyz': [0,0,0],
				'volume_absolute': mask_vol_gdth
			}
		print("Final")
		
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
	