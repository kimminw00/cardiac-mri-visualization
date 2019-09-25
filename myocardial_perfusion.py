'''

    2017-02-28

    place a colorbar.
'''

import sys
import scipy
import pickle
import scipy.misc

import numpy as np
import SimpleITK as sitk
import PyQt5
import pyqtgraph.opengl as gl
import matplotlib.pyplot as plt

import pyqtgraph as pg

from scipy import ndimage
from PyQt5 import QtCore, QtGui
# from pyqtgraph.Qt import QtGui

from pyqtgraph.pgcollections import OrderedDict

Gradients = OrderedDict([
	('bw', {'ticks': [(0.0, (0, 0, 0, 255)), (1, (255, 255, 255, 255))], 'mode': 'rgb'}),
	('hot', {'ticks': [(0.3333, (185, 0, 0, 255)), (0.6666, (255, 220, 0, 255)), (1, (255, 255, 255, 255)), (0, (0, 0, 0, 255))], 'mode': 'rgb'}),
	('jet', {'ticks': [(1, (166, 0, 0, 255)), (0.32247191011235954, (0, 255, 255, 255)), (0.11348314606741573, (0, 68, 255, 255)), (0.6797752808988764, (255, 255, 0, 255)), (0.902247191011236, (255, 0, 0, 255)), (0.0, (0, 0, 166, 255)), (0.5022471910112359, (0, 255, 0, 255))], 'mode': 'rgb'}),
	('summer', {'ticks': [(1, (255, 255, 0, 255)), (0.0, (0, 170, 127, 255))], 'mode': 'rgb'} ),
	('space', {'ticks': [(0.562, (75, 215, 227, 255)), (0.087, (255, 170, 0, 254)), (0.332, (0, 255, 0, 255)), (0.77, (85, 0, 255, 255)), (0.0, (255, 0, 0, 255)), (1.0, (255, 0, 127, 255))], 'mode': 'rgb'}),
	('winter', {'ticks': [(1, (0, 255, 127, 255)), (0.0, (0, 0, 255, 255))], 'mode': 'rgb'})
])


def crop(img, rect):
    '''
    :param img: image which i want to crop
    :param rect: [row_min, col_min, width, height]
    :return: cropped image
    '''
    row_min, col_min, width, height = rect
    new_img = img[row_min:row_min+height, col_min:col_min+width]
    return new_img


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

    cine_data = pickle.load(open(cine_pkl_file_name, 'rb'))
    lge_data = pickle.load(open(lge_pkl_file_name, 'rb'))

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

    numOfInsertedPicture = round(cine_spacingbetweenslices / cine_pixelspacing)

    mask_diastole_epi = mask_diastole_epi.astype(float)
    mask_diastole_endo = mask_diastole_endo.astype(float)

    mask_epi_3d = slice_interpolation(mask_diastole_epi[:, :, 3:8], numOfInsertedPicture)
    mask_endo_3d = slice_interpolation(mask_diastole_endo[:, :, 3:8], numOfInsertedPicture)

    moving_resampled_scar_255_inter_stack = []

    # scar interpolation
    _, _, loop_len = moving_resampled_scar_255_stack.shape
    for i in range(2, loop_len - 3):
        scar = moving_resampled_scar_255_stack[:, :, i + 1] > 0
        scar_before = moving_resampled_scar_255_stack[:, :, i] > 0
        scar = scar.astype(float)
        scar_before = scar_before.astype(float)
        Di_1 = scipy.ndimage.morphology.distance_transform_edt(
            scar_before) - scipy.ndimage.morphology.distance_transform_edt(1 - scar_before)
        Di = scipy.ndimage.morphology.distance_transform_edt(scar) - scipy.ndimage.morphology.distance_transform_edt(
            1 - scar)
        moving_resampled_scar_255_inter_stack.append(scar_before)
        for j in range(1, numOfInsertedPicture + 1):
            weight_Di = j / (numOfInsertedPicture + 1)
            weight_Di_1 = 1 - weight_Di
            image = weight_Di_1 * Di_1 + weight_Di * Di
            binary = image > 0
            binary = binary.astype(float)
            moving_resampled_scar_255_inter_stack.append(binary)
    moving_resampled_scar_255_inter_stack.append(scar)

    moving_resampled_scar_255_inter_stack = np.dstack(moving_resampled_scar_255_inter_stack)
    scar3d = moving_resampled_scar_255_inter_stack

    return mask_epi_3d, mask_endo_3d, scar3d


def planes_alignment(planes, zLVc, zRVi):

    aligned_planes = []
    num_of_surfaces = len(planes)

    for i in range(num_of_surfaces):
        idx = num_of_surfaces - i - 1
        xLVc, yLVc = zLVc[idx].real, zLVc[idx].imag
        xRVi, yRVi = zRVi[idx].real, zRVi[idx].imag
        planes[i].translate(-yLVc, -xLVc, 0)
        tan_val = (xRVi - xLVc) / (yRVi - yLVc)

        if i != 0:
            diff_angle = np.arctan((base_tan_val - tan_val) / (1 + base_tan_val*tan_val))
            planes[i].rotate(diff_angle * 180. / np.pi, 0, 0, 1)
        else:
            base_tan_val = tan_val

        aligned_planes.append(planes[i])

    return aligned_planes


def get_multiple_planes_nonbinary_colorbar(myo_map, slice_locations, zLVc, zRVi, ymin, ymax, cm):

    ret_planes = []
    row_num, col_num, num_of_surfaces = myo_map.shape

    midslice_loc_est = np.mean(slice_locations)

    for i in range(num_of_surfaces-1, -1, -1):

        xLVc, yLVc = zLVc[i].real, zLVc[i].imag
        xRVi, yRVi = zRVi[i].real, zRVi[i].imag

        upslope_values = myo_map[:, :, i]
        upslope_values_scaled = np.zeros(upslope_values.shape)

        for j in range(row_num):
            for k in range(col_num):
                upslope_values_scaled[j, k] = (upslope_values[j, k] - ymin) / (ymax - ymin)

        colors_rgba = cm.mapToFloat(upslope_values_scaled)
        for j in range(row_num):
            for k in range(col_num):
                if colors_rgba[j, k, 0] == 0 and colors_rgba[j, k, 1] == 0 and colors_rgba[j, k, 2] == 0:
                    colors_rgba[j, k, 3] = 0

        colors_rgba[int(yRVi), int(xRVi), :] = np.array([1., 1., 1., 1.])
        colors_rgba[int(yLVc), int(xLVc), :] = np.array([0., 1., 0., 1.])

        z = -(slice_locations[i] - midslice_loc_est) * np.ones((row_num, col_num))

        item = gl.GLSurfacePlotItem(z=z, colors=colors_rgba.reshape(row_num*col_num, 4), smooth=True, shader='balloon', glOptions='translucent')
        ret_planes.append(item)

    aligned_ret_planes = planes_alignment(ret_planes, zLVc, zRVi)

    return aligned_ret_planes


if __name__ == '__main__':
    # PyQt5.QtCore.QCoreApplication.addLibraryPath('.')
    # os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = 'D:\Anaconda3\envs\py34\Library\plugins\platforms'

    directory = "D:\\SimpleITK-Notebooks\\Python"
    sys.path.append(directory)

    perf_pkl_file_name = 'pkl_data/perfusion_1143____.pkl'

    perf_data = pickle.load(open(perf_pkl_file_name, 'rb'))

    upslope_map = perf_data['upslope_map']
    slice_locations = perf_data['slicelocation']
    zLVc = perf_data['LVc']
    zRVi = perf_data['RVi']

    # Create an PyQT4 application object.
    app = QtGui.QApplication(sys.argv)

    win = QtGui.QWidget()
    layout = QtGui.QGridLayout()
    win.setLayout(layout)
    cb = pg.GraphicsLayoutWidget()
    ax = pg.AxisItem('left'); ymin = 0.0; ymax = 0.4; # upslope range.
    ax.setRange(ymin, ymax)
    cb.addItem(ax)
    cmap = 'hot'
    gw = pg.GradientEditorItem(orientation='right'); GradientMode = Gradients[cmap]; gw.restoreState(GradientMode)
    cb.addItem(gw)
    view = gl.GLViewWidget()
    view.setSizePolicy(cb.sizePolicy())
    layout.addWidget(view, 0, 0)
    layout.addWidget(cb, 0, 1); layout.setColumnStretch(1, 0); layout.setColumnMinimumWidth(1, 120); layout.setColumnStretch(0, 1)
    view.sizeHint = lambda: pg.QtCore.QSize(1700, 800)
    cb.sizeHint = lambda: pg.QtCore.QSize(100, 800)
    layout.setHorizontalSpacing(0)

    win.resize(800, 800)

    cm = gw.colorMap()

    planes = get_multiple_planes_nonbinary_colorbar(upslope_map, slice_locations, zLVc, zRVi, ymin, ymax, cm)

    for plane in planes:
        view.addItem(plane)

    dst = 100
    view.setCameraPosition(distance=dst)

    win.show()

    print(slice_locations)

    cmap0 = 'hot'

    fig = plt.figure(3)
    for j in range(4):
        xLVc, yLVc = zLVc[j].real, zLVc[j].imag
        xRVi, yRVi = zRVi[j].real, zRVi[j].imag

        ax = plt.subplot(1, 4, j+1)
        cax = ax.imshow(upslope_map[:, :, j], cmap=cmap0, clim=[0, 0.4])
        plt.plot(xLVc, yLVc, 'ro')
        plt.plot(xRVi, yRVi, 'bo')

        ax.set_title('upslope map')
        cbar = fig.colorbar(cax, ticks=[0, 0.1, 0.2, 0.3, 0.4])
        cbar.ax.set_yticklabels(['0', '0.1', '0.2', '0.3', '0.4'])

    plt.show()

    sys.exit(app.exec_())
