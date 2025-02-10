import numpy as np
import cv2
from skimage.restoration import denoise_wavelet
from skimage.exposure import match_histograms
REF_IMAGE = np.load('ref_image.npy')

def adjust_intensity(image, increase=True, strength=50):
    if(increase):
        return np.clip(image + strength, 0, 65535) 
    else:
        return np.clip(image - strength, 0, 65535)

def gamma_correction(image, gamma= 1.0):
    normalized = image / 65535.0
    corrected = np.power(normalized, gamma)
    corrected = np.clip(corrected * 65535, 0, 65535).astype(np.uint16)
    return corrected

def apply_clahe(image, clip_limit=4.0, tile_grid_size=(3, 3)):
    zero_mask = image == 0
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image)
    enhanced[zero_mask] = 0
    return enhanced

def unsharp_masking(image, blur_kernel=(5, 5), alpha=1.5):
    blurred = cv2.GaussianBlur(image, blur_kernel, 0)
    return np.clip(image + alpha * (image - blurred), 0, 65535).astype(np.uint16)

def wavelet_denoising(image):
    return denoise_wavelet(image, rescale_sigma=True)

def bilateral_filter(image, d=9, sigma_color=50, sigma_space=50):
    image_float = image.astype(np.float32)
    filtered = cv2.bilateralFilter(image_float, d, sigma_color, sigma_space)
    return np.clip(filtered, 0, 65535).astype(np.uint16)

def match_intensity(image):
    return match_histograms(image, REF_IMAGE)

def denoise_image(image):
    image_8bit = np.uint8(image / 256)
    denoised_image_8bit = cv2.fastNlMeansDenoising(image_8bit, None, 10, 7, 21)
    denoised_image_16bit = np.uint16(denoised_image_8bit) * 256
    return denoised_image_16bit

# Process an input image according to it's vendor and output a processed image.
# It can work with any input dimensions and works best when the image has been masked using mask_image() function in helpers
def process_image(img, vendor):
    if(vendor == 'Planmed'):
        img = 65535-img
        img = img - 55535
        img = img/3.44
        img = np.uint16(img)

    elif(vendor == 'SIEMENS'):
        img = apply_clahe(img)

    elif(vendor == 'Siemens_INBreast'):
        img = apply_clahe(img)

    elif(vendor == 'DDSM'):
        img = img/23.948
        img = np.uint16(img)
        img = apply_clahe(img)
    
    elif(vendor in ['IMS s.r.l.', 'IMS GIOTTO S.p.A.', 'IMS']):
        _, thresholded = cv2.threshold(img, 0, 65535, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = cv2.bitwise_and(img, img, mask=thresholded.astype(np.uint8))
        img = np.where(thresholded == 0, 0, img)
        img = img/6.601
        img = np.uint16(img)
        img = apply_clahe(img, clip_limit=8.0, tile_grid_size=(3, 3))

    return img