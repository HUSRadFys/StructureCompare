import numpy as np
import pydicom
from skimage.draw import polygon
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
import seg_metrics.seg_metrics as sg
from tqdm import tqdm
import matplotlib.patches as mpatches
from pprint import pprint
from scipy import ndimage
from math import sqrt
import dicompylercore

class Structures:
	def __init__(self, folder, structureOrigin):
		self.folder = folder
		self.structureOrigin = structureOrigin
		self.ROIDict = dict()
		self.ROIIdxDict = dict()
		self.SOPInstanceUIDForStructureDict = dict()
		self.file = None
		self.filename = None
		self.uids = {}
		
		self.loadFile()
		self.findROINumbers()
		
	def loadFile(self):
		self.files = glob.glob(f"{self.folder}/{self.structureOrigin}/*.dcm")
		self.rsDict = { file : pydicom.dcmread(file) for file in self.files }
		self.file = list(self.rsDict.values())[0]
		self.filename = self.files[0]
		
	def getVendorName(self, rsFilename = None):
		if not rsFilename:
			rsFilename = list(self.rsDict.keys())[0]
		
		#vendorName = #rsFilename.split(".")[-2].split(" ")[0]
		vendorName = rsFilename.split(".")[0]
		
		return vendorName
	
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
					
			nameNumber = list()
			print(f"Found the following ROIs in the {self.folder} / {self.structureOrigin} dataset: ")
			for name, number in self.ROIDict[rsFilename].items():
				nameNumber.append(f"{name}")
			print(", ".join(nameNumber), end=".\n\n")
	
	def makeIndexForStructureName(self, structureName):
		for rsFilename, rs in self.rsDict.items():
			self.SOPInstanceUIDForStructureDict[rsFilename] = list()
			number = self.ROIDict[rsFilename][structureName.lower()]
			
			for idx, ROI in enumerate(rs.ROIContourSequence):
				if ROI.ReferencedROINumber == number:
					self.ROIIdxDict[rsFilename][structureName.lower()] = idx
					for cseq in ROI.ContourSequence:
						assert len(cseq.ContourImageSequence) == 1
						
						for ciseq in cseq.ContourImageSequence:
							UID = ciseq.ReferencedSOPInstanceUID
							self.SOPInstanceUIDForStructureDict[rsFilename].append(UID)
	
	def getListOfUIDs(self, rsFilename = None):
		if not rsFilename:
			rsFilename = list(self.rsDict.keys())[0]  
			
		return self.SOPInstanceUIDForStructureDict[rsFilename]
	
	def getListOfIdxSeq(self, rsFilename, SOPInstanceUID):
		idxSeqs = list()
		
		for idx, uid in enumerate(self.SOPInstanceUIDForStructureDict[rsFilename]):
			if uid == SOPInstanceUID:
				idxSeqs.append(idx)
		
		return idxSeqs
	
	def loadStructurePolygonAndMask(self, structureName, SOPInstanceUID, ds, rsFilename = None):
		if not rsFilename:
			rsFilename = list(self.rsDict.keys())[0]
			
		rs = self.rsDict[rsFilename]
		x0, y0 = ds.ImagePositionPatient[0:2]
		ps = ds.PixelSpacing
		#mask = np.zeros([ds.pixel_array.shape[0], ds.pixel_array.shape[1]])
		
		# Might be more than one idxSeq
		idxSeqs = self.getListOfIdxSeq(rsFilename, SOPInstanceUID)
		idxROI = self.ROIIdxDict[rsFilename][structureName.lower()]
		
		contours = list()
		masks = list()
		
		for idxSeq in idxSeqs:
			mask = np.zeros([512, 512])
			contourRaw = rs.ROIContourSequence[idxROI].ContourSequence[idxSeq].ContourData
			contour = np.reshape(contourRaw, (len(contourRaw) // 3, 3))
		
			contour_x = (contour[:, 0] - x0) / ps[0]
			contour_y = (contour[:, 1] - y0) / ps[1]
			contours.append([contour_x, contour_y])
	
			r, c = polygon(contour_x, contour_y, mask.shape)
			mask[r,c] = 1
			masks.append(mask)
			
		return contours, masks

	def makeListOfReferencedImages(self, folder):
		files = glob.glob(f"{folder}/CT*.dcm")
		for file in files:
			with pydicom.dcmread(file, stop_before_pixels=True) as ds:
				self.uids[ds.SOPInstanceUID] = ds.InstanceNumber - 1 # starts at 1

	def loadStructureMask3D(self, structureName, ds, rsFilename = None):
		if not rsFilename:
			rsFilename = list(self.rsDict.keys())[0]
			
		rs = self.rsDict[rsFilename]
		x0, y0 = ds.ImagePositionPatient[0:2]
		ps = ds.PixelSpacing
		#mask = np.zeros([ds.pixel_array.shape[0], ds.pixel_array.shape[1]])
		
		# Might be more than one idxSeq
		idxROI = self.ROIIdxDict[rsFilename][structureName.lower()]

		mask = np.zeros((512, 512, len(self.uids)), dtype=bool)

		print(f"The size of the mask for {structureName} is {np.shape(mask)}.")

		for cs in rs.ROIContourSequence[idxROI].ContourSequence:
			contourRaw = cs.ContourData
			idx = self.uids[cs.ContourImageSequence[0].ReferencedSOPInstanceUID] - 1

			contour = np.reshape(contourRaw, (len(contourRaw) // 3, 3))
		
			contour_x = (contour[:, 0] - x0) / ps[0]
			contour_y = (contour[:, 1] - y0) / ps[1]
	
			r, c = polygon(contour_x, contour_y, mask.shape)
			mask[r,c,idx] = 1
			
		return mask

	def get_roi(self, name):
		return list(self.ROIDict.values())[0][name]

	@property
	def structures(self):
		return list(self.ROIDict[list(self.rsDict.keys())[0]].keys())
	

folder = "zzART_Head1"

s_art = Structures(folder, "ART")
s_manual = Structures(folder, "Manual")

s_art.makeListOfReferencedImages(folder)
s_manual.makeListOfReferencedImages(folder)

ds = pydicom.dcmread(f"{folder}/CT.zzART_Head1.Image 1.dcm", stop_before_pixels=True)

common_structures = [ k for k in s_art.structures if k in s_manual.structures ]

print("Common structures")
# pprint(common_structures)

s = "spinalcord"
assert s in common_structures

s_art.makeIndexForStructureName(s)

mask_art = s_art.loadStructureMask3D(s, ds)

s_manual.makeIndexForStructureName(s)
mask_manual = s_manual.loadStructureMask3D(s, ds)

"""
spacing = list(ds.PixelSpacing) + [ds.SliceThickness]

to_use = [
	'dice',
	'jaccard',
	'hd',
	'hd95',
	'msd'
]

metrics = sg.write_metrics(labels=[1], gdth_img=mask_manual, pred_img=mask_art, csv_file=None,
                           spacing=spacing, metrics=to_use, verbose=True)

pprint(metrics)

"""

cm_art = ndimage.center_of_mass(mask_art)
cm_manual = ndimage.center_of_mass(mask_manual)

print("Center of mass manual")
pprint(cm_manual)

print("Center of mass ART")
pprint(cm_art)

diff = [k1-k2 for k1,k2 in zip(cm_art, cm_manual)]
euclid_diff = sqrt((diff[0] * ds.PixelSpacing[0])**2 + 
						 (diff[1] * ds.PixelSpacing[1])**2 +
						 (diff[2] * ds.SliceThickness)**2)
print(f"The euclidian center of mass difference is {euclid_diff:.2f} mm.")

voxel_size = ds.PixelSpacing[0] * ds.PixelSpacing[1] * ds.SliceThickness
sum_art = np.sum(mask_art) * voxel_size / (10**3)
sum_manual = np.sum(mask_manual) * voxel_size / (10**3)

print(f"The volume of ART is {sum_art:.1f} cc.")
print(f"The volume of Manual is {sum_manual:.1f} cc.")
print(f"The 3D VolumeDifference is {sum_art - sum_manual:.1f} cc.")

# What about DVH structure volume?
rs_file_art = s_art.filename
rs_file_manual = s_manual.filename
rd_file = glob.glob(s_art.folder + "/RD*")[0]

roi_number_art = s_art.get_roi(s)
roi_number_manual = s_manual.get_roi(s)

dvh_art = dicompyler.dvhcalc.get_dvh(rs_file_art, rd_file, roi_number_art)
dvh_manual = dicompyler.dvhcalc.get_dvh(rs_file_manual, rd_file, roi_number_manual)

volume_art = dvh_art.volume
volume_manual = dvh_manual.volume

print(f"The DVH volume of ART is {volume_art:.1f} cc.")
print(f"The DVH volume of Manual is {volume_manual:.1f} cc.")
print(f"The 3D VolumeDifference is {volume_art - volume_manual:.1f} cc.")
