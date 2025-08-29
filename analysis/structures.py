import numpy as np
import pydicom
from skimage.draw import polygon
from dicompylercore import dvhcalc
import glob
from pprint import pprint


class Structures:
	def __init__(self, folder, structureOrigin):
		self.structureOrigin = structureOrigin
		self.ROIDict = dict()
		self.ROIIdxDict = dict()
		self.folder = folder
		self.uids = {}
		self.volumes = dict()
		self.z = dict()
		self.x0y0 = dict()
		self.frame_of_reference_uid = {'CT': set(), 'RS': set()}
		
		self.loadFile()
		self.findROINumbers()
		self.makeListOfReferencedImages()

	def loadFile(self):
		self.files = glob.glob(f"{self.folder}/RS*.dcm")
		if self.structureOrigin == 'RS_groundtruth':
			self.files = [k for k in self.files if "GroundTruth" in k]
		else:
			self.files = [k for k in self.files if not "GroundTruth" in k]

		# self.rd_file = glob.glob(f"{self.folder}/RD*.dcm")[0]
		self.rsDict = { file : pydicom.dcmread(file) for file in self.files }

		for v in self.rsDict.values():
			self.frame_of_reference_uid['RS'].add(v.FrameOfReferenceUID)
	
	def getRsFilenames(self):
		return list(self.rsDict.keys())

	def findROINumbers(self):
		for rsFilename, rs in self.rsDict.items():
			self.ROIDict[rsFilename] = dict()
			self.ROIIdxDict[rsFilename] = dict()
			for ROI in rs.StructureSetROISequence:
				name = ROI.ROIName.lower()
				if name == "constrictmusc_pharynx":
					name = "pharynxconstrict"

				number = ROI.ROINumber
				self.ROIDict[rsFilename][name] = number

			for idx, ROI in enumerate(rs.ROIContourSequence):
				name = None
				for k,v in self.ROIDict[rsFilename].items():

					if v == ROI.ReferencedROINumber:
						name = k
						break
				assert name

				self.ROIIdxDict[rsFilename][name.lower()] = idx

	def makeListOfReferencedImages(self):
		files = glob.glob(f"{self.folder}/CT*.dcm")
		for file in files:
			with pydicom.dcmread(file, stop_before_pixels=True) as ds:
				self.uids[ds.SOPInstanceUID] = ds.InstanceNumber - 1 # starts at 1
				self.z[ds.SOPInstanceUID] = ds.ImagePositionPatient[2]
				self.x0y0[ds.SOPInstanceUID] = ds.ImagePositionPatient[:2]
				self.frame_of_reference_uid['CT'].add(ds.FrameOfReferenceUID)
				
				
	def loadStructureMask3D(self, structureName, ds, rsFilename = None, allowed_uids = list(), max_z = None):
		if not rsFilename:
			rsFilename = list(self.rsDict.keys())[0]
			
		rs = self.rsDict[rsFilename]
		ps = ds.PixelSpacing

		# Might be more than one idxSeq
		idxROI = self.ROIIdxDict[rsFilename][structureName.lower()]

		mask = np.zeros((ds.Columns, ds.Rows, len(self.uids)), dtype=bool)

		if not "ContourSequence" in rs.ROIContourSequence[idxROI]:
			print(f"No ContourSequence for {rsFilename} / {structureName}")
			return mask

		lastUID = list(self.x0y0.keys())[0]
		last_idx = None
		last_z = None

		for cs in rs.ROIContourSequence[idxROI].ContourSequence:
			contourRaw = cs.ContourData
			uid = cs.ContourImageSequence[0].ReferencedSOPInstanceUID
			try:
				x0, y0 = self.x0y0[uid]
			except:
				x0, y0 = self.x0y0[lastUID]
			#z0 = self.z[uid]
			z0 = contourRaw[2]

			if max_z and z0 < max_z:
				continue

			if not uid in self.uids:
				if last_z != z0:
					idx = last_idx - 1 # hack
				else:
					idx = last_idx # disconnected contours

			else:
				idx = self.uids[uid] # was - 1

			contour = np.reshape(contourRaw, (len(contourRaw) // 3, 3))
		
			# in pixel dimensions
			contour_x = (contour[:, 0] - x0) / ps[0]
			contour_y = (contour[:, 1] - y0) / ps[1]
	
			r, c = polygon(contour_x, contour_y, mask.shape)
			mask[r,c,idx] = 1
			lastUID = uid

			last_idx = idx
			last_z = z0

		return mask

	def get_roi(self, rsFilename, name):
		return self.ROIDict[rsFilename][name]

	def get_volume(self, structure_name, rs_file=None):
		if not rs_file:
			assert len(self.rs_files) == 1
			rs_file = self.rs_files[0]

		dvh = dvhcalc.get_dvh(
			rs_file, 
			self.rd_file,
			self.get_roi(rs_file, structure_name),
			interpolation_segments_between_planes=3
		)

		# in cc
		return dvh.volume

	def get_structures(self, rsFilename=None):
		if not rsFilename:
			assert len(self.rs_files) == 1
			rsFilename = self.rs_files[0]

		return list(self.ROIDict[rsFilename].keys())

	def make_index_z_list(self, rs, idxROI):
		# z_index = { uid : index }
		# 1) make uid : z value
		# 2) sort by z value; index by uid

		z_uid = dict()

		for cs in rs.ROIContourSequence[idxROI].ContourSequence:
			z = cs.ContourData[2]
			uid = cs.ContourImageSequence[0].ReferencedSOPInstanceUID
			z_uid[z] = uid

		sorted_z = sorted(list(z_uid.keys()))

		z_index = { z_uid[z]: idx for idx,z in enumerate(sorted_z) }
		return z_index

	@property
	def rs_files(self):
		return list(self.rsDict.keys())

	@property
	def ds(self):
		return self.ds
	