import numpy as np
import pydicom
from skimage.draw import polygon
import matplotlib.pyplot as plt
import glob

def findDICE(a,b):
    return np.sum(b[a==1])*2 / (np.sum(a) + np.sum(b))

class Images:
    def __init__(self, folder):
        self.folder = folder
        self.fileList = None
        # self.SOPInstanceUIDDict = dict()
        
        self.makeFileList()
        self.sortImagesAndMakeIndex()
        
    def makeFileList(self):
        self.fileList = glob.glob(f"{self.folder}/Images/*.dcm")
    
    def sortImagesAndMakeIndex(self):
        unsortedFileList = self.fileList
        
        sortingDict = dict()
        sortingDictSOP = dict()

                
        for file in unsortedFileList:
            ds = pydicom.dcmread(file, stop_before_pixels=True)
            sortingDict[ds.InstanceNumber] = file
            sortingDictSOP[ds.InstanceNumber] = ds.SOPInstanceUID
            #self.SOPInstanceUIDDict[ds.SOPInstanceUID] = file
            
        sortedKeys = sorted(list(sortingDict.keys()))
        sortedFileList = [ sortingDict[key] for key in sortedKeys ]
        
        
        self.SOPInstanceUIDDict = { sortingDictSOP[key] : sortingDict[key] for key in sortedKeys}
        self.SOPInstanceUIDList =  [ sortingDictSOP[key] for key in sortedKeys ]
        
        self.fileList = sortedFileList
        
    def getFileList(self):
        print(self.fileList)
        return self.fileList
    
    def getUIDList(self):
        return list(self.SOPInstanceUIDList)
        
    def loadDicom(self, SOPInstanceUID):
        
        filename = self.SOPInstanceUIDDict[SOPInstanceUID]
        ds = pydicom.dcmread(filename)
        return ds
    
    def loadDicomNoImage(self, SOPInstanceUID):
        filename = self.SOPInstanceUIDDict[SOPInstanceUID]
        ds = pydicom.dcmread(filename, stop_before_pixels=True)
        return ds
    
    def loadImage(self, SOPInstanceUID):
        ds = self.loadDicom(SOPInstanceUID)
        img = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        assert img.shape == (512, 512)
        return img
    
class Structures:
    def __init__(self, folder, structureOrigin):
        self.folder = folder
        self.structureOrigin = structureOrigin
        self.ROIDict = dict()
        self.ROIIdxDict = dict()
        self.SOPInstanceUIDForStructureDict = dict()
        self.file = None
        
        self.loadFile()
        self.findROINumbers()
        
    def loadFile(self):
        self.files = glob.glob(f"{self.folder}/Structures/{self.structureOrigin}/*.dcm")
        self.rsDict = { file : pydicom.dcmread(file) for file in self.files }
        
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
                
                if name == "bag_bowel":
                    name = "bowelbag"
                
                if not "ptv" in name and not "ctv" in name:
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
        ps = ds.PixelSpacing[0]
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
        
            contour_x = (contour[:, 0] - x0) / ps
            contour_y = (contour[:, 1] - y0) / ps
            contours.append([contour_x, contour_y])
    
            r, c = polygon(contour_x, contour_y, mask.shape)
            mask[r,c] = 1
            masks.append(mask)
            
        return contours, masks