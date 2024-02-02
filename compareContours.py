import matplotlib.pyplot as plt
import os
import numpy as np
from classes import Images, Structures
import pandas as pd
import seg_metrics.seg_metrics as sg
from tqdm import tqdm
import matplotlib.patches as mpatches

anatomicalSites = ['HeadAndNeck', 'Prostate', 'Breast']
anatomicalSite = anatomicalSites[1]
images = Images(anatomicalSite)

ww = 400
wl = 50

plt.figure(figsize=(8,8))

hiddenVendorNames = ['A','B','C','D','E','F','G','H','I','J','K','L','M']
vendorNames = ["MVision"]

hiddenVendorNamesDict = {vendorNames[k] : hiddenVendorNames[k] for k in range(len(vendorNames))}
 
structuresHUS = Structures(anatomicalSite, "GroundTruth")
structuresVendors = Structures(anatomicalSite, "Vendors")

with open(f"{anatomicalSite}/Output/namingKey.txt",'w') as keyFile:
    for vendorName, hiddenVendorName in hiddenVendorNamesDict.items():
        keyFile.write(f"Vendor {hiddenVendorName}: {vendorName}\n")

colors = {"Bladder": 'green', "Rectum": 'blue', 'Body' : 'red', "Bowelbag": "orange"}
structureNames = ["Bladder", "Rectum", "Bowelbag", "Body"]
# structureNames = ["Bladder"]

column_names = ["Slice number", "Structure", "Vendor", "DICE", "HD95", "MSD"]
df = pd.DataFrame(columns=column_names)

for structureName in structureNames:    
    print(f"Processing {structureName}...")
    structuresHUS.makeIndexForStructureName(structureName)    
    structuresVendors.makeIndexForStructureName(structureName)
    
    UIDs = list(set(structuresHUS.getListOfUIDs()))
    
    rsFilenames = structuresVendors.getRsFilenames()
        
    for thisUID in tqdm(UIDs):
        thisDs = images.loadDicomNoImage(thisUID)
        filenum = thisDs.InstanceNumber
        
        thisImg = images.loadImage(thisUID)
        plt.clf()
        plt.imshow(thisImg, cmap="gray", vmin=wl-ww/2, vmax=wl+ww/2)
        
        contoursHUS, masksHUS = structuresHUS.loadStructurePolygonAndMask(structureName, thisUID, thisDs)
        
        maskHUS = masksHUS[0]
        if len(masksHUS) > 1:               
            for idx, moreMaskHUS in enumerate(masksHUS[1:]):
                maskHUS = np.logical_xor(moreMaskHUS, maskHUS)
        
        for idx, contourHUS in enumerate(contoursHUS):
            label = "HUS"
            if idx > 0:
                label=None
            plt.plot(*contourHUS, color=colors[structureName], linestyle="solid", label=label)
        
        #plt.imshow(np.fliplr(np.rot90(maskHUS,k=3)))
        for rsFilename in rsFilenames:
            vendorName = structuresVendors.getVendorName(rsFilename)
            hiddenVendorName = hiddenVendorNamesDict[vendorName]
            vendorUIDs = list(set(structuresVendors.getListOfUIDs(rsFilename)))
            if not thisUID in vendorUIDs:
                continue
            
            contoursVendor, masksVendor = structuresVendors.loadStructurePolygonAndMask(structureName, thisUID, thisDs, rsFilename)
            
            maskVendor = masksVendor[0]
            if len(masksVendor) > 1:
                for moreMaskVendor in masksVendor[1:]:
                    maskVendor = np.logical_xor(maskVendor, moreMaskVendor)
            
            metrics = sg.write_metrics(labels=[1], gdth_img=maskHUS, pred_img=maskVendor, csv_file=None,
                                       spacing=thisDs.PixelSpacing, metrics=['dice','hd95','msd'], verbose=False)
            
            positive_difference = np.fliplr(np.rot90(np.where(maskHUS > maskVendor, 1, 0), k=3))
            negative_difference = np.fliplr(np.rot90(np.where(maskHUS < maskVendor, 1, 0), k=3))
                        
            DICE_sg = round(metrics[0]['dice'][0], 3)
            HD95_sg = round(metrics[0]['hd95'][0], 2)
            MSD_sg = round(metrics[0]['msd'][0],2)
            
            for idx, contourVendor in enumerate(contoursVendor):
                label = f"Vendor (DICE {DICE_sg:.3f}; HD95 {HD95_sg:.2f} mm)"
                if idx > 0:
                    label=None
                plt.plot(*contourVendor, color=colors[structureName], linestyle="dashed", label=label)
                
            #plt.imshow(np.fliplr(np.rot90(maskVendor, k=3)))
            plt.imshow(positive_difference, label="False Negative", cmap="Wistia", alpha=0.3*(positive_difference>0))
            plt.imshow(negative_difference, label="False Positive", cmap="spring", alpha=0.3*(negative_difference>0))
            
            new_row = {"Slice number": filenum, "Structure": structureName, "Vendor": vendorName, "DICE": DICE_sg, "HD95": HD95_sg, "MSD":MSD_sg}
            df = df.append(new_row, ignore_index=True)
        
            orange_patch = mpatches.Patch(color="orange", label="HUS not Vendor")
            yellow_patch = mpatches.Patch(color="yellow", label="Vendor not HUS")
            
            plt.title(f"{structureName}")
            l1 = plt.legend(loc=1)
            plt.legend(handles=[orange_patch, yellow_patch], loc=2)
            plt.gca().add_artist(l1)
            plt.tight_layout()
            plt.axis('off')
            
            path = f"{anatomicalSite}/Output/{hiddenVendorName}/{structureName}"
            if not os.path.exists(path):
                os.makedirs(path)                
            plt.savefig(f"{path}/{filenum}.png")
        
df = df.sort_values(["Structure", "Vendor", "Slice number"])        
df.to_csv(f"{anatomicalSite}/Output/indices.csv", sep=";", decimal=",")

plt.figure()

#df.
