"""
计算centroid
"""
import nibabel as nib
import os
import numpy as np

datapath="crop_resize"

list=[11,12,13,14,15,16,17,18,
      21,22,23,24,25,26,27,28,
      31,32,33,34,35,36,37,38,
      41,42,43,44,45,46,47,48]

# vol_path = []
seg_paths = []
datalist=os.listdir(datapath)
print(datalist)
for item in datalist:
    vol_temp_path = 'volume-' + item + '.nii.gz'
    seg_temp_path = 'segmentation-' + item + '.nii.gz'
    # vol_path.append(os.path.join(datapath, item, vol_temp_path))
    seg_paths.append(os.path.join(datapath, item, seg_temp_path))

for step,seg_path in enumerate(seg_paths):
    seg = nib.load(seg_paths[0]).get_fdata()
    print(seg_path)

    centroid_x=[]
    centroid_y=[]
    centroid_z=[]

    centroidmap_x=seg.copy()
    centroidmap_y=seg.copy()
    centroidmap_z=seg.copy()

    for label in list:
        
        
        x,y,z=np.where(seg==label)
        centroid_x.append(np.sum(x)//x.size)
        centroid_y.append(np.sum(y)//y.size)
        centroid_z.append(np.sum(z)//z.size)

    # print(centroid_x)
    assert len(list)==len(centroid_x)
    # out_seg=np.zeros((256,256,256))

    for i in range(len(list)):
        centroidmap_x[centroidmap_x==list[i]]=centroid_x[i]
        centroidmap_y[centroidmap_y==list[i]]=centroid_y[i]
        centroidmap_z[centroidmap_z==list[i]]=centroid_z[i]
        # out_seg[centroid_x[i]-1:centroid_x[i]+1,centroid_y[i]-1:centroid_y[i]+1,centroid_z[i]-1:centroid_z[i]+1]=list[i]

    outx = nib.Nifti1Image(centroidmap_x, affine=np.eye(4))
    outpath='centroid_mapx-'+datalist[step] +'.nii.gz'
    print(os.path.join(datapath,datalist[step],outpath))
    nib.save(outx,os.path.join(datapath,datalist[step],outpath) )

    outy = nib.Nifti1Image(centroidmap_y, affine=np.eye(4))
    outpath='centroid_mapy-'+datalist[step] +'.nii.gz'
    print(os.path.join(datapath,datalist[step],outpath))
    nib.save(outy,os.path.join(datapath,datalist[step],outpath) )

    outz = nib.Nifti1Image(centroidmap_z, affine=np.eye(4))
    outpath='centroid_mapz-'+datalist[step] +'.nii.gz'
    print(os.path.join(datapath,datalist[step],outpath))
    nib.save(outz,os.path.join(datapath,datalist[step],outpath) )

    # out = nib.Nifti1Image(out_seg, affine=np.eye(4))
    # outpath='centroid_map-'+datalist[step] +'.nii.gz'
    # print(os.path.join(datapath,datalist[step],outpath))
    # nib.save(out,os.path.join(datapath,datalist[step],outpath) )