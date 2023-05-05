import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from scipy.interpolate import interp1d, interp2d, griddata
from scipy.ndimage.filters import gaussian_filter
import math
import shelve


def interporlate_weight_map(xq, yq, x, y, val, kind):
    interp_f = interp2d(x, y, val, kind=kind)
    return interp_f(xq, yq)

def centeredDistanceMatrix(n):
    # make sure n is odd
    x,y = np.meshgrid(range(n),range(n))
    return np.sqrt( (x-(n-1)/2)**2 + (y-(n-1)/2)**2)

def nd_interp(d,y,n):
    x = np.arange(n) 
    f = interp1d(x, y, bounds_error=False, fill_value = 0)
    return f(d.flat).reshape(d.shape)

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def arbitrary_SFR_adjust(imager_SFR, enhancement=1):
    num_samples = imager_SFR.shape[0]
    num_roi = imager_SFR.shape[1]
    imager_SFR = np.multiply( imager_SFR,
                              np.multiply(np.reshape( np.linspace(1,enhancement,num_samples), (num_samples,1)), 
                                          np.reshape( np.linspace(1,1,num_roi), (1,num_roi)))
                                    )
    return imager_SFR

def foveated_blurring(sample_num, roi_num):
    # arbitrary mtf foveation effect
    imager_SFR = np.multiply(np.reshape( np.exp(-np.square(np.arange(0,sample_num))/2e2), (sample_num,1)), 
                                         np.reshape( np.linspace(1,1,roi_num), (1,roi_num)))
    imager_SFR[:,:8] = np.multiply(np.reshape( np.exp(-np.square(np.arange(0,sample_num))/2e4), (sample_num,1)), 
                                   np.reshape( np.linspace(1,1,8), (1,8)))
    return imager_SFR

def uniform_Gaussian_blurring(sample_num, roi_num, sigma):
    imager_SFR = np.multiply(np.reshape( np.exp(-np.square(np.arange(0,sample_num))/sigma), (sample_num,1)), 
                                         np.reshape( np.linspace(1,1,roi_num), (1,roi_num)))
    return imager_SFR

def random_Gaussian_blurring(sample_num, roi_num, sigma):
    imager_SFR = np.zeros((sample_num, roi_num))
    np.random.seed(1)
    sigmas = sigma + 1.8*sigma*(np.random.rand(roi_num) - 0.5)
    for i in range(roi_num):
        imager_SFR[:,i] = np.exp(-np.square(np.arange(0,sample_num))/sigmas[i])
    return imager_SFR

def load_scene_image(scene_name):
    scene_image = imread(scene_name)
    scene_image_pixel = 1.4e-3 # [mm]
    scene_image_EFL = 4.2 # [mm], effective focal length (EFL) of the lens that took the picture
    h, w, c = scene_image.shape
    return scene_image[:, int((w/2)-(h/2)):int((w/2)+(h/2)), :], scene_image_pixel, scene_image_EFL

def center_crop(img, dim):
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img

def image_centered_crop_or_pad(img, target_size):
    i_h = img.shape[0]
    i_w = img.shape[1]
    o_h = target_size[0]
    o_w = target_size[1]
    if i_h > o_h:
        img = center_crop(img, (o_h, i_w))
    else:
        img = np.pad(img, [(int((o_h-i_h)/2), int((o_h-i_h)/2)),(0, 0)], mode='constant', constant_values=0)
    if i_w > o_w:
        img = center_crop(img, (o_h, o_w))
    else:
        img = np.pad(img, [(0,0), (int((o_w-i_w)/2), int((o_w-i_w)/2))], mode='constant', constant_values=0)
    return img
        
def image_system_analysis(imager_SFR, n_dimensions=3):
    n = imager_SFR.shape[1]
    roi_num = imager_SFR.shape[2]

    A = np.zeros((2*n-1, roi_num))
    for idx in range(roi_num):
        mtf_1d = nd_interp(np.array([i for i in range(n-1,0,-1)]+[i for i in range(n)]),
                           imager_SFR[1,:,idx],n)
        psf_1d = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(mtf_1d))).real # center the psf, psf_1d should be real since mtf_1d is symmetric
        psf_1d = psf_1d/np.sum(psf_1d) # normalize by psf energy (value already intensity)
        A[:,idx] = np.reshape(psf_1d, (2*n-1,))
        
    # -- PCA -- #
    A_avg = np.mean(A, axis=1, keepdims=1)
    u, s, vh = np.linalg.svd(A - A_avg, full_matrices=False)
    eig_psf = np.append(u[:,0:n_dimensions], A_avg, axis=1) 
    eig_psf_coef_at_roi = np.append(vh[0:n_dimensions, :], np.ones((1,vh.shape[1])), axis=0)
    psf_x = (1/imager_SFR[0,-1,0])*np.arange(-n+1,n)
    _, axs_1 = plt.subplots(2,3, figsize=(6, 6))
    axs_1[0,0].plot(imager_SFR[0,:,0], imager_SFR[1,:,:])
    axs_1[0,0].set_title('MTF at each ROI')
    axs_1[0,0].set_ylabel('MTF score')
    axs_1[0,0].set_xlabel('spatial frequency [lp/mm]')
    axs_1[0,1].plot(psf_x, A)
    axs_1[0,1].set_title('PSF at each ROI')
    axs_1[0,1].set_xlabel('x [mm]')
    axs_1[0,2].plot(psf_x, eig_psf)
    axs_1[0,2].set_title('eigen PSF')
    axs_1[0,2].set_xlabel('x [mm]')
    axs_1[1,0].imshow(u)
    axs_1[1,0].set_title('U')
    axs_1[1,1].plot(s)
    axs_1[1,1].set_title('s \n the weight of each eigen PSF')
    axs_1[1,2].imshow(vh)
    axs_1[1,2].set_title('V.T')
    plt.tight_layout()
    s = np.append(s[0:n_dimensions], 1)
    return eig_psf, eig_psf_coef_at_roi, s

def eigen_MTF(eig_psf, eig_psf_coef_at_roi, pad_width, scene_h, scene_w, px, py, qX, qY, SFR_map_h, SFR_map_w):
    n = int((eig_psf.shape[0] + 1)/2)
    num_eigen = eig_psf.shape[1] - 1
    d = centeredDistanceMatrix(2*n-1)
    mtf_img = np.zeros((scene_h, scene_w, num_eigen + 1))
    weight_map = np.zeros((scene_h, scene_w, num_eigen + 1))
    for i in range(num_eigen + 1):
        mtf_1d = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(eig_psf[:,i]))).real # ignore imag, which should be negligible due to symmetric psf
        mtf_2d = nd_interp(d, mtf_1d[n-1:], n)
        # calibrate the physical unit of mtf spatial freqs (to fit img F), pad zeros and interpolate
        mtf_2d = np.pad(mtf_2d, pad_width, pad_with, padder = 0)
        mtf_img[:,:,i] = cv2.resize(mtf_2d.T, dsize=(scene_h, scene_w), interpolation=cv2.INTER_CUBIC)

        # 2D interpolate the spatial-variability of eigen-psf, ai(x,y)
        weight_interpolated = griddata((px, py), eig_psf_coef_at_roi[i,:], (qX, qY), method='linear', fill_value = 0)
        weight_img_scaled = cv2.resize(weight_interpolated, 
                                       dsize=(2*int(SFR_map_w/2), 2*int(SFR_map_h/2)), 
                                       interpolation=cv2.INTER_CUBIC)
        weight_map[:,:,i] = image_centered_crop_or_pad(weight_img_scaled, (scene_h, scene_w))
    return mtf_img, weight_map

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def normalize_image(img):
    return img/np.max(img)

def compare_images(final_image, scene_image, scene_cy_per_rad):
    scene_h, scene_w, scene_c = scene_image.shape
    view_angle_range = [(x*(180/math.pi)*math.atan((0.5*scene_h)*(1/scene_cy_per_rad))) for x in [-1, 1, -1, 1]]
    
    _, axs_3 = plt.subplots(1,2, figsize=(16, 8))
    axs_3[0].imshow(scene_image, extent = np.array(view_angle_range))
    axs_3[0].set_title('original image')
    axs_3[0].set_xlabel('viewing angle [deg]')
    axs_3[1].imshow(final_image, extent = np.array(view_angle_range))
    axs_3[1].set_title('processed image')
    axs_3[1].set_xlabel('viewing angle [deg]')

def main():
# -- load variables -- #
    bk_restore = shelve.open('./demo_data.pkl')
    imager_SFR = bk_restore['imager_SFR']
    SFR_coordinates = bk_restore['SFR_coordinates']
    SFR_coordinates_map_size = bk_restore['SFR_coordinates_map_size']
    bk_restore.close()

    num_SFR_point = imager_SFR.shape[1]
    num_ROI = imager_SFR.shape[2]
    SFR_map_h, SFR_map_w = SFR_coordinates_map_size[0], SFR_coordinates_map_size[1]


# -- arbitrary imager SFR specification at corresponding spatial coordinates -- #
    example_idx = 1
    if example_idx == 1:
        scene_name = './portrait.jpg'
        imager_SFR[1,:,:] = foveated_blurring(num_SFR_point, num_ROI)
    else:
        scene_name = './star_night.jpg'
        imager_SFR[1,:,:] = random_Gaussian_blurring(num_SFR_point, num_ROI, sigma = 1e4)
    # imager_SFR[1,:,:] = uniform_Gaussian_blurring(num_SFR_point, num_ROI, sigma = 1e4)
    # imager_SFR[1,:,:] = arbitrary_SFR_adjust(imager_SFR[1,:,:], enhancement = 1.8)


# -- PCA on spatial-variant SFR -- #
    num_eigen = 5
    eig_psf, eig_psf_coef_at_roi, s = image_system_analysis(imager_SFR, n_dimensions=num_eigen)


# -- load input image -- #
    scene_image, scene_image_pixel, scene_image_EFL = load_scene_image(scene_name)
    scene_h, scene_w, scene_c = scene_image.shape
    scene_cy_per_rad = scene_image_EFL/scene_image_pixel
    image_fourier = np.fft.fftshift(np.fft.fft2(scene_image[:,:,0]))
    _, axs_2 = plt.subplots(3,1 + num_eigen + 1, figsize=(14, 5))
    axs_2[1,0].imshow(scene_image/np.max(scene_image))
    axs_2[0,0].imshow(np.log(np.abs(image_fourier)),
                      extent =[-scene_cy_per_rad, scene_cy_per_rad, -scene_cy_per_rad, scene_cy_per_rad])
    axs_2[0,0].set_title('scene spectrum')
    axs_2[0,0].set_ylabel('ky [cy/rad]')
    axs_2[0,0].set_xlabel('kx [cy/rad]')


# -- calculate eigen MTF of eigen PSF and ai(u,v) -- #
    px = np.array([loc[0] for loc in SFR_coordinates])
    py = np.array([loc[1] for loc in SFR_coordinates])
    qx = np.arange(1, SFR_map_w +1)
    qy = np.arange(1, SFR_map_h +1)
    qX, qY = np.meshgrid(qx, qy)

    pad_width = int( ((num_SFR_point/(imager_SFR[0,-1,0]*scene_image_EFL)) * scene_cy_per_rad) - num_SFR_point) # lens MTF should convert in cy/rad space rathan than lp/mm space
    
    mtf_img, ai = eigen_MTF(eig_psf, eig_psf_coef_at_roi, pad_width, scene_h, scene_w, px, py, qX, qY, SFR_map_h, SFR_map_w)


# -- PSF blurring -- #
    # for extending the synthesize FOV outside the range of specified SFR coordinates, but using just average PSF
    pos = np.where(ai[:,:,num_eigen]==0)
    ai[pos[0], pos[1], num_eigen] = 1


    blurred_img = np.zeros((scene_h, scene_w, num_eigen + 1, scene_c))
    for i in range(num_eigen + 1):
        for ch in range(scene_c):
            # (I.*ai) conv pi = iff( fft{I.*ai} x eigen-mtf )
            weighted_image_fourier = np.fft.fftshift(np.fft.fft2(np.multiply( scene_image[:,:,ch], ai[:,:,i] )))
            blurred_img_spec = np.multiply(weighted_image_fourier, mtf_img[:,:,i])
            # transform back to real space for each MTF mode
            blurred_img[:,:,i,ch] = s[i]*np.fft.ifft2(np.fft.ifftshift(blurred_img_spec)).real # img should be real since F is symmetric

        axs_2[0,i+1].imshow(mtf_img[:,:,i])
        axs_2[0,i+1].set_title('eigen MTF ' + str(i))
        axs_2[0,i+1].set_axis_off()
        axs_2[1,i+1].imshow(ai[:,:,i])
        axs_2[1,i+1].set_title('weight map ' + str(i))
        axs_2[1,i+1].set_axis_off()
        axs_2[2,i+1].imshow(blurred_img[:,:,i,:]/np.max(blurred_img[:,:,i,:])) # x, y, eigen, ch
        axs_2[2,i+1].set_title('blurring with ' + str(i))
        axs_2[2,i+1].set_axis_off()
        axs_2[2,0].axis('off')
    axs_2[0,-1].set_title('avg MTF ')
    axs_2[1,-1].set_title('weight map ')
    axs_2[2,-1].set_title('blurring with avg')
    plt.tight_layout()
    
    # sum of blurred images by different eigen psf
    final_image = np.average(blurred_img, axis=2)
    final_image = normalize_image(final_image)
    compare_images(final_image, scene_image, scene_cy_per_rad)
    i=1


if __name__ == "__main__":
    main()

