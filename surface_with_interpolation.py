import sys
import scipy
import numpy as np
import pyqtgraph.opengl as gl

from scipy import ndimage
from skimage import measure
from PyQt4 import QtGui


def setdiff_mask(mask1, mask2):
    nx = mask1.shape[0]
    ny = mask1.shape[1]
    mask1_ = mask1.reshape((nx*ny, 1))
    mask2_ = mask2.reshape((nx*ny, 1))
    ind1 = np.where(mask1_ == 1)
    ind2 = np.where(mask2_ == 1)

    index1 = ind1[0]
    index2 = ind2[0]

    s1 = set(index1)
    s2 = set(index2)
    s3 = s1 - s2

    mask3 = np.zeros((nx, ny))

    for j in s3:
        mask3[int(j/ny), j%ny] = 1

    return mask3

result = np.load('contour_results_2.npy')

print(result.shape)

cine_images = result[0]  # cine는 4차원영상

print(cine_images.shape)  # 272: 행 갯수, 232:열 , 30:시간측(갯수) , 9: 슬라이스갯수

dicom_info = result[1]

print(dicom_info.PixelSpacing)  # 픽셀실제크기 mm
print(dicom_info.SpacingBetweenSlices)  # mm 10/1.28 배 z축 슬라이스 늘림

mask_diastole_endo = result[2]  # 확장기mask : binary
mask_diastole_epi = result[3]
frameno_diastole = result[4]  # diastole일때  idx
slicelocation = result[5]  # 위치 : 크기차이 == SpacingBetweenSlices

print(slicelocation[0:4])
print(mask_diastole_endo.shape)
print(frameno_diastole)

start_idx = 0
mask_diastole = []
row, col, sliceNum = mask_diastole_endo.shape
const = round(dicom_info.SpacingBetweenSlices / dicom_info.PixelSpacing[0])

for slno in range(start_idx, sliceNum-1):
    mask2 = mask_diastole_epi[:, :, slno]
    mask1 = mask_diastole_endo[:, :, slno]
    mask3 = setdiff_mask(mask2, mask1)

    # assume that len(mask_diastole) >= 2
    if slno !=  start_idx :
        Di_1 = ndimage.morphology.distance_transform_edt(mask3_before)
        Di = ndimage.morphology.distance_transform_edt(mask3)
        for i in range(1, const):
            weight_Di_1 = i / const
            weight_Di = 1 - weight_Di_1
            mask_diastole.append( weight_Di_1 * mask3_before + weight_Di * mask3 )
    mask3_before = mask3
    mask_diastole.append( mask3 )

mask_diastole = np.dstack(mask_diastole)

print(mask_diastole.shape)

# Create an PyQT4 application object.
app = QtGui.QApplication(sys.argv)

# Create a window object.
window = gl.GLViewWidget()
window.resize(500, 500)
window.setCameraPosition(distance=100)
window.setWindowTitle('pyqtgraph : GLIsosurface')
window.show()

# uniform_filter() is equivalent to smooth3() in matlab.
mask_diastole = scipy.ndimage.uniform_filter(mask_diastole, [5, 5, 20], mode='nearest')

# Using marching cubes algorithm to get a polygonal mesh of an isosurface
verts, faces = measure.marching_cubes(mask_diastole, 0.1)
meshdata = gl.MeshData(vertexes=verts, faces=faces)
mesh = gl.GLMeshItem(meshdata=meshdata, smooth=True, color=(1.0, 0.0, 0.0, 0.2), shader='balloon', glOptions='additive')

# Translation
[avgX, avgY, avgZ] = map(np.mean, zip(*verts))
mesh.translate(-avgX, -avgY, -avgZ)
window.addItem(mesh)

sys.exit(app.exec_())

plt.show()

sys.exit()
