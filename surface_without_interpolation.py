import sys
import scipy
import numpy as np
import pyqtgraph.opengl as gl

from scipy import io
from skimage import measure
from PyQt4 import QtGui

# Create an PyQT4 application object.
app = QtGui.QApplication(sys.argv)

# Create a window object.
window = gl.GLViewWidget()
window.resize(500, 500)
window.setCameraPosition(distance=100)
window.setWindowTitle('pyqtgraph : GLIsosurface')
window.show()

# Read data from a mat file.
mat_file = io.loadmat('segment_myo4_frame30.mat')
myo3d = mat_file['myo3d']

# uniform_filter() is equivalent to smooth3() in matlab.
myo3d = scipy.ndimage.uniform_filter(myo3d, [5, 5, 20], mode='nearest')

# Using marching cubes algorithm to get a polygonal mesh of an isosurface
verts, faces = measure.marching_cubes(myo3d, 0.1)
meshdata = gl.MeshData(vertexes=verts, faces=faces)
mesh = gl.GLMeshItem(meshdata=meshdata, smooth=True, color=(1.0, 0.0, 0.0, 0.2), shader='balloon', glOptions='additive')

# Translation
[avgX, avgY, avgZ] = map(np.mean, zip(*verts))
mesh.translate(-avgX, -avgY, -avgZ)
window.addItem(mesh)

sys.exit(app.exec_())
