from analysis.patient import Patient
from analysis.export import to_excel

folder = "data/zzART_Head1"

patient = Patient(folder)
metrics = patient.build_metrics()
to_excel(metrics, 'zzART_Head1.xlsx')


"""
s_groundtruth = Structures(folder, "groundtruth")
s_compare = Structures(folder, "compare")

s_groundtruth.makeListOfReferencedImages(folder)
s_compare.makeListOfReferencedImages(folder)

ds = pydicom.dcmread(f"{folder}/CT.zzgroundtruth_Head1.Image 1.dcm", stop_before_pixels=True)

common_structures = [ k for k in s_groundtruth.structures if k in s_compare.structures ]

s = "spinalcord"

s_groundtruth.makeIndexForStructureName(s)
mask_groundtruth = s_groundtruth.loadStructureMask3D(s, ds)

s_compare.makeIndexForStructureName(s)
mask_compare = s_compare.loadStructureMask3D(s, ds)

voxel_size = ds.PixelSpacing[0] * ds.PixelSpacing[1] * ds.SliceThickness
sum_groundtruth = np.sum(mask_groundtruth) * voxel_size / (10**3)
sum_compare = np.sum(mask_compare) * voxel_size / (10**3)

print(f"The volume of groundtruth is {sum_groundtruth:.1f} cc.")
print(f"The volume of compare is {sum_compare:.1f} cc.")
print(f"The 3D VolumeDifference is {sum_groundtruth - sum_compare:.1f} cc.")

# What about DVH structure volume?
rs_file_groundtruth = s_groundtruth.filename
rs_file_compare = s_compare.filename
rd_file = glob.glob(s_groundtruth.folder + "/RD*")[0]

roi_number_groundtruth = s_groundtruth.get_roi(s)
roi_number_compare = s_compare.get_roi(s)

dvh_groundtruth = dicompylercore.dvhcalc.get_dvh(rs_file_groundtruth, rd_file, interpolate_between_slices=3)
dvh_compare = dicompylercore.dvhcalc.get_dvh(rs_file_compare, rd_file, roi_number_compare, interpolate_between_slices=3)

volume_groundtruth = dvh_groundtruth.volume
volume_compare = dvh_compare.volume

print(f"The DVH volume of groundtruth is {volume_groundtruth:.1f} cc.")
print(f"The DVH volume of compare is {volume_compare:.1f} cc.")
print(f"The 3D VolumeDifference is {volume_groundtruth - volume_compare:.1f} cc.")
"""