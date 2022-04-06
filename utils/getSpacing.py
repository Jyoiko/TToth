import os
import SimpleITK as sitk
from utils import crop,resize_image_itk
datapath = "../data"
vol_temp_path = os.listdir(os.path.join(datapath, "imagesTr"))
seg_temp_path = os.listdir(os.path.join(datapath, "labelsTr"))
for path in vol_temp_path:
    img=sitk.ReadImage(os.path.join(datapath, "imagesTr",path))
    seg = sitk.ReadImage(os.path.join(datapath, "labelsTr", path))
    # img,seg=crop(img,seg)
    img=resize_image_itk(img,(192,192,192))
    print(img.GetSpacing())
