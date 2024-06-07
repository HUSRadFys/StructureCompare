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
		
		self.loadFile()
		self.findROINumbers()
		self.makeListOfReferencedImages()

	def loadFile(self):
		self.files = glob.glob(f"{self.folder}/{self.structureOrigin}/*.dcm")
		self.rd_file = glob.glob(f"{self.folder}/RD*.dcm")[0]
		self.rsDict = { file : pydicom.dcmread(file) for file in self.files }
	
	def getRsFilenames(self):
		return list(self.rsDict.keys())

	def findROINumbers(self):
		for rsFilename, rs in self.rsDict.items():
			self.ROIDict[rsFilename] = dict()
			self.ROIIdxDict[rsFilename] = dict()
			for ROI in rs.StructureSetROISequence:
				name = ROI.ROIName.lower()
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

	def loadStructureMask3D(self, structureName, ds, rsFilename = None):
		if not rsFilename:
			rsFilename = list(self.rsDict.keys())[0]
			
		rs = self.rsDict[rsFilename]
		x0, y0 = ds.ImagePositionPatient[0:2]
		ps = ds.PixelSpacing
		
		# Might be more than one idxSeq
		idxROI = self.ROIIdxDict[rsFilename][structureName.lower()]

		mask = np.zeros((ds.Columns, ds.Rows, len(self.uids)), dtype=bool)

		for cs in rs.ROIContourSequence[idxROI].ContourSequence:
			contourRaw = cs.ContourData
			idx = self.uids[cs.ContourImageSequence[0].ReferencedSOPInstanceUID] - 1

			contour = np.reshape(contourRaw, (len(contourRaw) // 3, 3))
		
			# in pixel dimensions
			contour_x = (contour[:, 0] - x0) / ps[0]
			contour_y = (contour[:, 1] - y0) / ps[1]
	
			r, c = polygon(contour_x, contour_y, mask.shape)
			mask[r,c,idx] = 1
			
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

	@property
	def rs_files(self):
		return list(self.rsDict.keys())

	@property
	def ds(self):
		return self.ds
	