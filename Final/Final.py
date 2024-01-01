#這傳錯的檔案
import pydicom
import matplotlib.pyplot as plt
import numpy as np
#dicom
dicom_file_path = './Data/DICOM_DLCSI033/CT.1.2.840.35235.2019100318100201428817886189951872608741.79.dcm'

ds = pydicom.dcmread(dicom_file_path)
image = ds.pixel_array
## convert to HU
image = ds.RescaleSlope * image + ds.RescaleIntercept

## lung
min_value_lung = -1000
max_value_lung = 400
image_lung = image.clip(min_value_lung, max_value_lung)
## bone
min_value_bone = 200
max_value_bone = 2000
image_bone = image.clip(min_value_bone, max_value_bone)


## plot
plt.figure(figsize=(14, 4))
# original
plt.subplot(1, 3, 1)
plt.title("original")
plt.pcolormesh(image, cmap='Greys_r')
plt.colorbar(label='HU')
plt.axis('off')
# lung
plt.subplot(1, 3, 2)
plt.title("lung")
plt.pcolormesh(image_lung, cmap='Greys_r')
plt.colorbar(label='HU')
plt.axis('off')
#bone
plt.subplot(1, 3, 3)
plt.title("bone")
plt.pcolormesh(image_bone, cmap='Greys_r')
plt.colorbar(label='HU')
plt.axis('off')

plt.tight_layout()
plt.show()
