'''
    2017-03-02

    add a colorbar.

    GLVolumeItem
    GLMeshItem
'''

import time

import sys
import scipy
import pickle
import warnings
import numpy as np
import scipy.spatial
import SimpleITK as sitk
import matplotlib.patches
import PyQt5
import pyqtgraph.opengl as gl

from PyQt5 import QtCore, QtWidgets
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt

import pyqtgraph as pg

from scipy import ndimage
from skimage import measure
from pyqtgraph.Qt import QtGui

from pyqtgraph.pgcollections import OrderedDict

Gradients = OrderedDict([
    ('bw', {'ticks': [(0.0, (0, 0, 0, 255)), (1, (255, 255, 255, 255))], 'mode': 'rgb'}),
    ('hot', {'ticks': [(0.3333, (185, 0, 0, 255)), (0.6666, (255, 220, 0, 255)), (1, (255, 255, 255, 255)), (0, (0, 0, 0, 255))], 'mode': 'rgb'}),
    ('jet', {'ticks': [(1, (166, 0, 0, 255)), (0.32247191011235954, (0, 255, 255, 255)), (0.11348314606741573, (0, 68, 255, 255)), (0.6797752808988764, (255, 255, 0, 255)), (0.902247191011236, (255, 0, 0, 255)), (0.0, (0, 0, 166, 255)), (0.5022471910112359, (0, 255, 0, 255))], 'mode': 'rgb'}),
    ('summer', {'ticks': [(1, (255, 255, 0, 255)), (0.0, (0, 170, 127, 255))], 'mode': 'rgb'} ),
    ('space', {'ticks': [(0.562, (75, 215, 227, 255)), (0.087, (255, 170, 0, 254)), (0.332, (0, 255, 0, 255)), (0.77, (85, 0, 255, 255)), (0.0, (255, 0, 0, 255)), (1.0, (255, 0, 127, 255))], 'mode': 'rgb'}),
    ('winter', {'ticks': [(1, (0, 255, 127, 255)), (0.0, (0, 0, 255, 255))], 'mode': 'rgb'})
])

FIGURE = False


class GLViewWidget_with_text(gl.GLViewWidget):

    '''
    coord = [anterior_coord, septal_coord, inferior_coord, lateral_coord]
    z = text height
    '''

    def __init__(self, coord, z):
        gl.GLViewWidget.__init__(self)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.setFont(font)
        self.coord = coord
        self.z = z

    def paintGL(self, *args, **kwds):
        gl.GLViewWidget.paintGL(self, *args, **kwds)
        self.qglColor(QtCore.Qt.white)
        for i in range(4):
            if i == 0:
                text = 'anterior'
            elif i == 1:
                text = 'septal'
            elif i == 2:
                text = 'inferior'
            elif i == 3:
                text = 'lateral'
            x, y = self.coord[i]
            self.renderText(y, x, self.z, text)


def read_lge_pkl(lge_pkl_filename):

    SLNO = 6

    with open(lge_pkl_filename, 'rb') as pkl_file:
        lge_pkl_data = pickle.load(pkl_file)
        nSD_N = lge_pkl_data['nSD_N']
        lge_img = lge_pkl_data['lge_img']
        lge_epi_mask = lge_pkl_data['mask_epi']
        lge_endo_mask = lge_pkl_data['mask_endo']
        scar_nSD_mask = lge_pkl_data['mask_scar_nSD']
        lge_scar_mask = scar_nSD_mask[:, :, :, nSD_N - 2]
        lge_pixelspacing = lge_pkl_data['pixelspacing']
        lge_spacingbetweenslices = lge_pkl_data['spacingbetweenslices']

    numOfInsertedPicture = int(lge_spacingbetweenslices[0, 0] / lge_pixelspacing[0, 0])

    lge_epi_mask = slice_interpolation(lge_epi_mask[:, :, 3:8], numOfInsertedPicture)
    lge_endo_mask = slice_interpolation(lge_endo_mask[:, :, 3:8], numOfInsertedPicture)
    lge_scar_mask = slice_interpolation(lge_scar_mask[:, :, 2:7], numOfInsertedPicture)

    return lge_img[:, :, SLNO], lge_epi_mask, lge_endo_mask, lge_scar_mask


def get_text_position(img):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title("Where are the anterior, septal, inferior and lateral?")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coord = plt.ginput(4, timeout=-1)
    plt.show(block=False)
    plt.close()
    return coord


def get_xyposition_fromMask(mask):
    position = np.where(mask == 1)
    return position


def diff_mask(lhs_mask, rhs_mask):

    """
    :param lhs_mask : 2d binary numpy array
    :param rhs_mask : 2d binary numpy array
    :return : lhs_mask - rhs_mask (vectorized operation)
    """
    row_num, col_num = lhs_mask.shape
    ret_mask = np.zeros(lhs_mask.shape)

    for i in range(row_num):
        for j in range(col_num):
            if lhs_mask[i, j] == 1 and rhs_mask[i, j] == 0:
                ret_mask[i, j] = 1

    return ret_mask


def find_com(image_array):
    """"
    find a center of mass of polynomial
    :param image_array: 2d image(numpy.ndarray)
    :return: a center of mass of polynomial
    """

    sitk_image = sitk.GetImageFromArray(image_array, isVector=True)

    ''' Region growing to segment the LV blood pool'''

    plt.figure()
    plt.imshow(image_array, cmap='gray')
    plt.title('locate LV center point')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x1 = plt.ginput(1, timeout=-1)
    plt.show(block=False)
    plt.close()

    seed = (int(x1[0][0]), int(x1[0][1]))

    seg_rg = sitk.ConnectedThreshold(sitk_image, seedList=[seed], lower=300, upper=650)

    mask_rg = sitk.GetArrayFromImage(seg_rg)
    mask_endo = np.zeros(image_array.shape)
    mask_final = np.zeros(image_array.shape)
    mask_endo[mask_rg > 0] = 1

    ''' convex hull to enclose the papillary muscles in endocardial border detection '''

    points = []
    for x in range(mask_endo.shape[0]):
        for y in range(mask_endo.shape[1]):
            if mask_endo[x][y] > 0.0:
                points.append([y, x])

    hull = scipy.spatial.ConvexHull(points)
    position = []
    for idx in hull.vertices:
        position.append(points[idx])
    poly = matplotlib.patches.Polygon(position, animated=True, alpha=0.3)
    for x in range(image_array.shape[1]):
        for y in range(image_array.shape[0]):
            if poly.get_path().contains_point((x, y)):
                mask_final[y][x] = 1
            else:
                mask_final[y][x] = 0

    mask_endo_final = mask_final

    xypos = get_xyposition_fromMask(mask_endo_final)

    ypos = xypos[0]
    xpos = xypos[1]

    com_x = np.mean(xpos)  # x center
    com_y = np.mean(ypos)  # y center

    plt.figure()
    plt.title('LV center point')
    plt.imshow(image_array, cmap='gray')
    plt.plot(com_x, com_y, c='b', marker='s', alpha=1.0)
    plt.show()

    return com_x, com_y


def closest_point(point, points):
    points = np.asarray(points)
    dist_2 = np.sum((points-point)**2, axis=1)
    return np.argmin(dist_2)


def slice_interpolation(mask, num_of_inserted_picture):

    mask_diastole_myo = []
    _, _, slice_num = mask.shape

    for i in range(0, slice_num-1):
        mask1 = mask[:, :, i+1] > 0
        mask1_1 = mask[:, :, i] > 0
        mask1 = mask1.astype(float)
        mask1_1 = mask1_1.astype(float)

        Di_1_mask1 = scipy.ndimage.morphology.distance_transform_edt(mask1_1) - scipy.ndimage.morphology.distance_transform_edt(1 - mask1_1)
        Di_mask1 = scipy.ndimage.morphology.distance_transform_edt(mask1) - scipy.ndimage.morphology.distance_transform_edt(1 - mask1)

        mask_diastole_myo.append(mask1_1)
        for j in range(1, num_of_inserted_picture + 1):
            weight_Di = j / (num_of_inserted_picture + 1)
            weight_Di_1 = 1 - weight_Di
            image_1 = weight_Di_1 * Di_1_mask1 + weight_Di * Di_mask1
            binary_1 = image_1 > 0
            binary_1 = binary_1.astype(float)
            mask_diastole_myo.append(binary_1)

    mask_diastole_myo.append(mask1)
    mask_diastole_myo = np.dstack(mask_diastole_myo)
    return mask_diastole_myo


def crop(img, rect):

    '''
    :param img: image which i want to crop
    :param rect: [row_min, col_min, width, height]
    :return: cropped image
    '''
    row_min, col_min, width, height = rect
    new_img = img[row_min:row_min+height, col_min:col_min+width]
    return new_img


def partition(image, myoseg_mask, scar_mask, com_x, com_y, slice_num, n_segments):

    """
    :param image:
    :param com_yx: center of mass, x: column 방향, y: row 방향
    :param n_segments:
    :return:
    """

    max_y, max_x = myoseg_mask.shape

    if FIGURE:
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.plot(com_x, com_y, c='r', marker='s', alpha=1.0)
        plt.contour(myoseg_mask, [0.5], colors='r')
        plt.contour(scar_mask, [0.5], colors='b')

    transmurality = []
    nsamp = 500

    R = np.minimum(max_y, max_x)
    z0 = complex(com_x, com_y)

    for ind in range(n_segments):

        z1 = z0 + R * np.exp(-1j*2*np.pi*ind/n_segments)

        x = np.linspace(z0.real, z1.real, nsamp)
        y = np.linspace(z0.imag, z1.imag, nsamp)

        z = scipy.ndimage.map_coordinates(myoseg_mask, np.vstack((y, x)))

        epsilon = 0.05

        for p in range(z.shape[0]):
            if np.abs(z[p]-1) < epsilon:
                endo_point_idx = p
                break

        for q in range(z.shape[0]-1, 0, -1):
            if np.abs(z[q]-1) < epsilon:
                epi_point_idx = q
                break

        num_of_scar_points = 0

        if endo_point_idx >= epi_point_idx:
            continue

        for p in range(endo_point_idx, epi_point_idx):
            if scar_mask[int(y[p]), int(x[p])] > 0:
                num_of_scar_points += 1

        transmural_val = num_of_scar_points / (epi_point_idx - endo_point_idx + 1)

        x2 = x[epi_point_idx]
        y2 = y[epi_point_idx]

        transmurality.append(((y2, x2, slice_num), transmural_val))

        if FIGURE:
            plt.plot(x[endo_point_idx], y[endo_point_idx], c='y', marker='o', alpha=1.0)
            plt.plot(x[epi_point_idx], y[epi_point_idx], c='g', marker='o', alpha=1.0)
            plt.annotate(str(int(transmural_val*100)), xy=(x2, y2))

    if FIGURE:
        plt.show()

    return transmurality


def registration(cine_pkl_file_name, lge_pkl_file_name):

    """
    :param cine_pkl_file_name: cine pickle file name
    :param lge_pkl_file_name: lge pickle file name
    :return:
        mask_epi_3d :
        mask_endo_3d :
        scar3d :
    """

    SLNO = 6

    with open(cine_pkl_file_name, 'rb') as cine_file:
        cine_data = pickle.load(cine_file)

    with open(lge_pkl_file_name, 'rb') as lge_file:
        lge_data = pickle.load(lge_file)

    lge_img = lge_data['lge_img']
    nSD_N = lge_data['nSD_N']
    mask_scar_nSD = lge_data['mask_scar_nSD']
    mask_myoseg = lge_data['mask_myoseg']

    cine_img = cine_data['cine_img']
    mask_diastole_endo_cine = cine_data['mask_diastole_endo']
    mask_diastole_epi_cine = cine_data['mask_diastole_epi']
    frameno_diastole = cine_data['frameno_diastole']
    cine_pixelspacing = cine_data['pixelspacing']
    cine_spacingbetweenslices = cine_data['spacingbetweenslices']

    # if it is True, then lge we will use scipy.misc.imresize.
    # if it is False, we will not use use scipy.misc.imresize.
    lge_resize_flag = True

    nrow, ncol, _, loop_len = cine_img.shape

    mask_scar_nSD_img_stack = []

    _, _, loop_len, _ = mask_scar_nSD.shape

    if lge_resize_flag:
        # interpolation
        ret = scipy.misc.imresize(lge_img[:, :, SLNO], 1.56 / cine_pixelspacing, interp='bicubic')
        row_min = 30
        col_min = 30
        cropped_lge_img = crop(ret, [row_min, col_min, ncol, nrow])

        mask_myoseg_img = scipy.misc.imresize(mask_myoseg[:, :, SLNO], 1.56 / cine_pixelspacing, interp='bicubic')

        for i in range(loop_len):
            mask_scar_nSD_img = scipy.misc.imresize(mask_scar_nSD[:, :, i, nSD_N - 2], 1.56 / cine_pixelspacing,
                                                    interp='bicubic')
            cropped_mask_scar_nSD_img = crop(mask_scar_nSD_img, [row_min, col_min, ncol, nrow])
            mask_scar_nSD_img_stack.append(cropped_mask_scar_nSD_img)

        mask_scar_nSD_img = scipy.misc.imresize(mask_scar_nSD[:, :, SLNO, nSD_N - 2], 1.56 / cine_pixelspacing,
                                                interp='bicubic')
    else:
        ret = scipy.misc.imresize(lge_img[:, :, SLNO], 1.56 / cine_pixelspacing, interp='bicubic')
        cropped_lge_img = ret

    fixed_image = cine_img[:, :, frameno_diastole, SLNO]
    moving_image = cropped_lge_img

    fixed_image = sitk.GetImageFromArray(fixed_image, isVector=True)
    moving_image = sitk.GetImageFromArray(moving_image, isVector=True)

    fixed_image_255 = sitk.Cast(sitk.RescaleIntensity(fixed_image), sitk.sitkUInt8)
    moving_image_255 = sitk.Cast(sitk.RescaleIntensity(moving_image), sitk.sitkUInt8)

    initial_transform = sitk.CenteredTransformInitializer(fixed_image_255,
                                                          moving_image_255,
                                                          sitk.Euler2DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)

    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    learning_rate = 1.0  # default = 1.0
    registration_method.SetOptimizerAsGradientDescent(learningRate=learning_rate, numberOfIterations=100,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Connect all of the observers so that we can perform plotting during registration.
    final_transform = registration_method.Execute(sitk.Cast(fixed_image_255, sitk.sitkFloat32),
                                                  sitk.Cast(moving_image_255, sitk.sitkFloat32))

    moving_resampled_scar_255_stack = []

    for i in range(len(mask_scar_nSD_img_stack)):
        mask_scar_nSD_255 = sitk.GetImageFromArray(mask_scar_nSD_img_stack[i], isVector=True)
        mask_scar_nSD_255 = sitk.Cast(sitk.RescaleIntensity(mask_scar_nSD_255), sitk.sitkUInt8)
        moving_resampled_scar = sitk.Resample(mask_scar_nSD_255, fixed_image_255, final_transform, sitk.sitkLinear, 0.0,
                                              mask_scar_nSD_255.GetPixelIDValue())
        moving_resampled_scar_255 = sitk.GetArrayFromImage(moving_resampled_scar)
        moving_resampled_scar_255_stack.append(moving_resampled_scar_255)

    moving_resampled_scar_255_stack = np.dstack(moving_resampled_scar_255_stack)


    mask_diastole_endo = mask_diastole_endo_cine
    mask_diastole_epi = mask_diastole_epi_cine


    mask_diastole_epi = mask_diastole_epi.astype(float)
    mask_diastole_endo = mask_diastole_endo.astype(float)

    numOfInsertedPicture = round(cine_spacingbetweenslices / cine_pixelspacing)

    mask_epi_3d = slice_interpolation(mask_diastole_epi[:, :, 3:8], numOfInsertedPicture)
    mask_endo_3d = slice_interpolation(mask_diastole_endo[:, :, 3:8], numOfInsertedPicture)
    scar3d = slice_interpolation(moving_resampled_scar_255_stack[:, :, 2:7], numOfInsertedPicture)

    return cine_img[:, :, frameno_diastole, SLNO], mask_epi_3d, mask_endo_3d, scar3d


def get_transmurality_data_colorbar(verts, mask_epi_3d, mask_endo_3d, cm):

    _, _, loop_len = mask_epi_3d.shape
    global_transmurality = []

    num_radial_line = 360
    n_neighbor = 26  # 6 or 26

    for slno in range(loop_len):
        all_points = np.where(mask_endo_3d[:, :, slno] == 1)
        com_y = np.mean(all_points[0])
        com_x = np.mean(all_points[1])
        myoseg_mask = diff_mask(mask_epi_3d[:, :, slno], mask_endo_3d[:, :, slno])
        myoseg_mask = myoseg_mask > 0
        myoseg_mask.astype(int)
        global_transmurality += partition(None, myoseg_mask, scar3d[:, :, slno], com_x, com_y, slno, num_radial_line)

    volume_data = np.zeros(mask_diastole.shape + (4,), dtype=np.ubyte)

    nvertex, _ = verts.shape

    int_verts_tup = tuple(map(tuple, verts.astype(int)))

    for i in range(nvertex):
        volume_data[int_verts_tup[i]] = [0, 0, 0, 255]

    for tup in global_transmurality:
        t_value = tup[1]  # transmurality value. range: 0 - 1

        colors_rgba = cm.mapToFloat(t_value)

        idx = closest_point(tup[0], verts)
        y, x, z = int_verts_tup[idx]
        volume_data[y, x, z] = [255 * colors_rgba[0], 255 * colors_rgba[1], 255 * colors_rgba[2], 255 * colors_rgba[3]]

        if n_neighbor == 6:
            for i in range(3):
                for j in -1, 1:
                    try:
                        if i == 0 and not np.array_equal(volume_data[y+j, x, z], np.array([0, 0, 0, 0])):
                            volume_data[y+j, x, z] = [255 * colors_rgba[0], 255 * colors_rgba[1], 255 * colors_rgba[2], 255 * colors_rgba[3]]
                        if i == 1 and not np.array_equal(volume_data[y, x+j, z], np.array([0, 0, 0, 0])):
                            volume_data[y, x+j, z] = [255 * colors_rgba[0], 255 * colors_rgba[1], 255 * colors_rgba[2], 255 * colors_rgba[3]]
                        if i == 2 and not np.array_equal(volume_data[y, x, z+j], np.array([0, 0, 0, 0])):
                            volume_data[y, x, z+j] = [255 * colors_rgba[0], 255 * colors_rgba[1], 255 * colors_rgba[2], 255 * colors_rgba[3]]
                    except IndexError:
                        pass
        elif n_neighbor == 26:
            for i in range(1, -2, -1):
                for j in range(1, -2, -1):
                    for k in range(1, -2, -1):
                        try:
                            if not np.array_equal(volume_data[y+i, x+j, z+k], np.array([0, 0, 0, 0])):
                                volume_data[y+i, x+j, z+k] = [255 * colors_rgba[0], 255 * colors_rgba[1], 255 * colors_rgba[2], 255 * colors_rgba[3]]
                        except IndexError:
                            pass

    return volume_data


def get_facecolors_transmurality(faces, verts, mask_epi_3d, mask_endo_3d, cm):
    '''
    use this function when using GLMeshItem
    :param verts:
    :param mask_epi_3d:
    :param mask_endo_3d:
    :param cm:
    :return: facecolors
    '''

    _, _, loop_len = mask_epi_3d.shape
    global_transmurality = []
    num_radial_line = 180
    for slno in range(loop_len):
        all_points = np.where(mask_endo_3d[:, :, slno] == 1)
        com_y = np.mean(all_points[0])
        com_x = np.mean(all_points[1])
        myoseg_mask = diff_mask(mask_epi_3d[:, :, slno], mask_endo_3d[:, :, slno])
        myoseg_mask = myoseg_mask > 0
        myoseg_mask.astype(int)
        global_transmurality += partition(None, myoseg_mask, scar3d[:, :, slno], com_x, com_y, slno, num_radial_line)

    facecolors = np.zeros((faces.shape[0], 4))

    nfaces, _ = faces.shape

    is_painted_face = np.zeros(nfaces)

    for i in range(nfaces):
        facecolors[i, :] = [0., 0., 0., 1.]

    for tup in global_transmurality:
        t_value = tup[1]  # transmurality value. range: 0 - 1
        colors_rgba = cm.mapToFloat(t_value)
        idx = closest_point(tup[0], verts)
        for j in range(nfaces):
            if idx in faces[j]:
                facecolors[j, :] = colors_rgba
                is_painted_face[j] = 1

    for i in range(1, nfaces):
        if is_painted_face[i] == 0:
            facecolors[i, :] = facecolors[i-1, :]

    return facecolors


if __name__ == '__main__':

    FIGURE = False
    only_read_lge_pkl = True  # lge_only == False means we register cine and LGE.

    start_time = time.time()
    if only_read_lge_pkl:
        myoseg_2d_mask, mask_epi_3d, mask_endo_3d, scar3d = read_lge_pkl('pkl_data/LGE_scar_1143____new.pkl')
    else:
        myoseg_2d_mask, mask_epi_3d, mask_endo_3d, scar3d = registration('pkl_data/cine_lv_1143____.pkl', 'pkl_data/LGE_scar_1143____new.pkl')

    end_time = time.time()

    print('registration : ', end_time-start_time)

    text_pos = get_text_position(myoseg_2d_mask)

    start_time = time.time()
    # uniform_filter() is equivalent to smooth3() in matlab.
    mask_diastole = scipy.ndimage.uniform_filter(mask_epi_3d, [4, 4, 20], mode='nearest')
    # Using marching cubes algorithm to get a polygonal mesh of an isosurface
    verts, faces = measure.marching_cubes_classic(mask_diastole, level=0.5)
    end_time = time.time()
    print('marching_cubes : ', end_time-start_time)

    [avgX, avgY, avgZ] = map(np.mean, zip(*verts))
    [avgX_text, avgY_text] = map(np.mean, zip(*text_pos))
    text_pos = list(map(lambda p: (p[0] - avgX_text, p[1] - avgY_text), text_pos))

    app = QtWidgets.QApplication(sys.argv)

    # vis3D_method = 'glmeshitem'   # 'glvolumeitem' or 'glmeshitem'
    vis3D_method = 'glmeshitem'

    if vis3D_method == 'glvolumeitem':

        win = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout()
        win.setLayout(layout)
        cb = pg.GraphicsLayoutWidget()
        ax = pg.AxisItem('left')
        ymin, ymax = 0.0, 1.0  # scar transmurality range.
        ax.setRange(ymin, ymax)
        cb.addItem(ax)
        cmap = 'jet'
        gw = pg.GradientEditorItem(orientation='right'); GradientMode = Gradients[cmap]; gw.restoreState(GradientMode)
        cb.addItem(gw)

        view = GLViewWidget_with_text(text_pos, 0)
        view.setSizePolicy(cb.sizePolicy())
        layout.addWidget(view, 0, 0)
        layout.addWidget(cb, 0, 1)
        layout.setColumnStretch(1, 0)
        layout.setColumnMinimumWidth(1, 120)
        layout.setColumnStretch(0, 1)
        view.sizeHint = lambda: pg.QtCore.QSize(1700, 800)
        cb.sizeHint = lambda: pg.QtCore.QSize(100, 800)
        layout.setHorizontalSpacing(0)

        cm = gw.colorMap()

        start_time = time.time()
        volume_data = get_transmurality_data_colorbar(verts, mask_epi_3d, mask_endo_3d, cm)
        end_time = time.time()
        print('marching_cubes : ', end_time - start_time)

        slden = 20  # 20

        volume_data = gl.GLVolumeItem(volume_data, sliceDensity=slden, smooth=False)
        volume_data.translate(-avgX, -avgY, -avgZ)
        view.addItem(volume_data)

        dst = 100
        view.setCameraPosition(distance=dst)

        win.show()

    elif vis3D_method == 'glmeshitem':

        print(vis3D_method)
        applicationDefinition = QtWidgets.QApplication(sys.argv)
        win = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout()
        win.setLayout(layout)
        cb = pg.GraphicsLayoutWidget()
        ax = pg.AxisItem('left')
        ymin, ymax = 0.0, 1.0  # scar transmurality range.
        ax.setRange(ymin, ymax)
        cb.addItem(ax)
        cmap = 'jet'
        gw = pg.GradientEditorItem(orientation='right')
        GradientMode = Gradients[cmap]
        gw.restoreState(GradientMode)
        cb.addItem(gw)

        view = gl.GLViewWidget()
        view.setSizePolicy(cb.sizePolicy())
        layout.addWidget(view, 0, 0)
        layout.addWidget(cb, 0, 1)
        layout.setColumnStretch(1, 0)
        layout.setColumnMinimumWidth(1, 120)
        layout.setColumnStretch(0, 1)
        view.sizeHint = lambda: pg.QtCore.QSize(1700, 800)
        cb.sizeHint = lambda: pg.QtCore.QSize(100, 800)
        layout.setHorizontalSpacing(0)

        cm = gw.colorMap()

        facecolors = get_facecolors_transmurality(faces, verts, mask_epi_3d, mask_endo_3d, cm)

        print('verts:', verts.shape)
        print('faces:', faces.shape)
        print('facecolors:', facecolors.shape)

        vis3d = gl.GLMeshItem(vertexes=verts, faces=faces, faceColors=facecolors, drawEdges=False, smooth=False)
        vis3d.translate(-avgX, -avgY, -avgZ)
        view.addItem(vis3d)

        dst = 100
        view.setCameraPosition(distance=dst)

        win.show()
        applicationDefinition.exec_()

    sys.exit(app.exec_())
