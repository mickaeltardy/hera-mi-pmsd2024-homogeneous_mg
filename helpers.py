import pandas as pd
import os as os
import numpy as np
from skimage import feature as skfeat
from scipy import stats as scipy_stats
import cv2
from skimage.restoration import denoise_wavelet
from skimage.exposure import match_histograms
REF_IMAGE = np.load('ref_image.npy')


def get_ddsm_table(base_path):
    ddsm_files = [base_path+"/mass_case_description_test_set.csv",
    base_path+"/calc_case_description_test_set.csv",
    base_path+"/calc_case_description_train_set.csv",
    base_path+"/mass_case_description_train_set.csv"]

    ddsm_dfs = []
    for f in ddsm_files:
        ddsm_dfs.append(pd.read_csv(f))
    ddsm_df = pd.concat(ddsm_dfs)
    ddsm_df['image_prefix'] = ddsm_df['image file path'].apply(lambda x: x.split('/')[0])

    local_files = []
    try:
        with os.scandir(base_path) as entries:
            for entry in entries:
                if entry.is_dir():
                    local_files.append(entry.name)
    except Exception:
        pass

    # local_files = [entry.name for entry in os.scandir("./DDSM/") if entry.is_dir()]
    ddsm_df = ddsm_df[ddsm_df['image_prefix'].isin(local_files)]
    ddsm_df = ddsm_df[ddsm_df['image_prefix'].map(ddsm_df['image_prefix'].value_counts()) == 1]
    file_names = [os.listdir(os.path.join(base_path, ddsm_df.iloc[i]['image_prefix']))[0] for i in range(len(ddsm_df))]
    ddsm_df['full_path'] = [os.path.join(base_path, ddsm_df.iloc[i]['image_prefix'], file_names[i]) for i in range(len(file_names))]
    ddsm_df['Manufacturer'] = 'DDSM'

    return ddsm_df

def get_INBreast_table(base_path):
    xlsdata = pd.read_csv(os.path.join(base_path,'INbreast.csv'), sep=';')
    biradsclass = xlsdata['Bi-Rads'].values
    biradsclassFilename = xlsdata['File Name'].values
    date = xlsdata['Acquisition date'].values

    sorted_indices = np.argsort(biradsclassFilename)
    biradsclass = biradsclass[sorted_indices]
    date = date[sorted_indices]

    dicom_files = [f for f in os.listdir(os.path.join(base_path,'AllDICOMs')) if f.endswith('.dcm')]
    nn = len(dicom_files)
    INbreast = np.empty((nn, 8), dtype=object)

    for k in range(nn):
        # info = pydicom.dcmread(dicom_files[k])
        # INbreast[k, 0] = k + 1
        
        # Split filename
        lineData = dicom_files[k].split('_')
        # INbreast[k, 1] is empty - name
        INbreast[k, 1] = lineData[1]  # patient ID
        INbreast[k, 2] = lineData[2]  # Modality - MG
        INbreast[k, 3] = lineData[3]  # Left or Right Breast
        INbreast[k, 4] = lineData[4]  # ViewPosition CC or ML(O)
        # INbreast[k, 5] is empty - date
        INbreast[k, 6] = dicom_files[k]  # filename
        # INbreast[k, 7] is empty - birads

    dicomFilename = INbreast[:, 6]
    sorted_indices = np.argsort(dicomFilename)
    INbreast = INbreast[sorted_indices]
    INbreast[:, 7] = biradsclass
    INbreast[:, 5] = date
    
    INbreast_cases = pd.DataFrame(INbreast)
    INbreast_cases.columns = ['Patient_Name', 'Patient_ID', 'Modality', 'Left or Right Breast', 'Image View', 'Date', 'file_name', 'BIRADS']
    INbreast_cases['file_name_prefix'] = INbreast_cases['file_name'].apply(lambda x: x.split('_')[0])
    xlsdata['File Name'] = xlsdata['File Name'].apply(str)
    INbreast_cases = INbreast_cases.merge(xlsdata[['File Name', 'ACR']], how='inner', left_on='file_name_prefix', right_on='File Name')
    INbreast_cases.drop(columns=['file_name_prefix'], inplace=True)
    INbreast_cases['full_path'] = [os.path.join(base_path,"AllDICOMs",INbreast_cases.iloc[i]['file_name']) for i in range(len(INbreast_cases))]
    INbreast_cases['Manufacturer'] = 'Siemens_INBreast'

    return INbreast_cases

def get_VinDR_table(base_path):
    vinDR_files = [base_path+'/metadata.csv', base_path+'/breast-level_annotations.csv', base_path+'/finding_annotations.csv']
    metadata = pd.read_csv(vinDR_files[0])
    breast_annotations = pd.read_csv(vinDR_files[1])
    finding_annotations = pd.read_csv(vinDR_files[2])

    local_files = []
    try:
        with os.scandir(base_path) as entries:
            for entry in entries:
                if entry.is_dir():
                    local_files.append(entry.name)
    except Exception:
        pass
    # local_files = [entry.name for entry in os.scandir("./VinDR/") if entry.is_dir()]
    breast_annotations = breast_annotations[breast_annotations['study_id'].isin(local_files)]
    finding_annotations = finding_annotations[finding_annotations['study_id'].isin(local_files)]
    annotations = breast_annotations.merge(finding_annotations, how='inner', on=['study_id', 'series_id', 'image_id'], suffixes=('_b', '_f'))
    annotations.drop(columns=['laterality_f', 'view_position_f', 'breast_birads_f', 'breast_density_f', 'split_f', 'height_f', 'width_f'], inplace=True)

    metadata = metadata[metadata['Series Instance UID'].isin(annotations['series_id'].unique())]
    metadata.drop(columns=['SOP Instance UID.1'], inplace=True)
    metadata = metadata.rename(columns={"SOP Instance UID": "image_id", "Series Instance UID": "series_id"})

    vindr_df = annotations.merge(metadata, how='inner', on=['series_id', 'image_id'], suffixes=('_a', '_m'))

    vindr_df = vindr_df.rename(columns={'laterality_b': 'laterality', 
                 'view_position_b': 'view_position', 
                 'height_b': 'height', 
                 'width_b': 'width', 
                 'breast_birads_b': 'breast_birads', 
                 'breast_density_b': 'breast_density',
                 'split_b': 'split'})
    
    vindr_df['full_path'] = [os.path.join(base_path,vindr_df.iloc[i]['study_id'],vindr_df.iloc[i]['image_id']+'.dicom') for i in range(len(vindr_df))]

    return vindr_df

def normalize_img(image):
    image = image.astype(np.float32)
    nonzero_pixels = image[image > 0]
    min_val = np.min(nonzero_pixels)
    max_val = np.max(nonzero_pixels)
    
    normalized_image = (image - min_val) / (max_val - min_val)
    normalized_image[image == 0] = 0  # Keep zero pixels as zero
    return normalized_image

def run_intensity_functions(img):

    img = normalize_img(img)
    mean = np.mean(img[img>0.])
    max = float(np.max(img))
    min = float(np.min(img[img>0.])) 
    std = np.std(img[img>0.])
    median = np.median(img[img>0.])
    skew = scipy_stats.skew(img[img>0.])
    kurtosis = scipy_stats.kurtosis(img[img>0.])
    
    num_bins = 16
    histogram, bins = np.histogram(img[img > 0.], bins=num_bins)
    area_under_histogram = np.trapz(histogram)

    hist_dict = {}
    for i in range(num_bins):
        hist_dict.update({f'hist_{i+1}': histogram[i], f'bin_{i+1}':bins[i]})

    return {
        'mean': mean,
        'max': max,
        'min': min,
        'std': std,
        'median': median,
        'skew': skew,
        'kurtosis': kurtosis,
        'area_under_histogram': area_under_histogram,
    }, hist_dict

def mask_image(pixel_array):

    non_zero_mask = pixel_array > 0
    contours, _ = cv2.findContours(
        non_zero_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    largest_contour = max(contours, key=cv2.contourArea)
    breast_mask = np.zeros_like(pixel_array, dtype=pixel_array.dtype)
    cv2.drawContours(breast_mask, [largest_contour], -1, 1, thickness=cv2.FILLED)
    breast_tissue_image = (pixel_array * breast_mask).astype(pixel_array.dtype)
    return breast_tissue_image

def compute_glcm(img):
    img8bit = img.copy()
    img8bit_min = img8bit[img8bit>0].min()
    img8bit_max = img8bit[img8bit>0].max()
    img8bit[img8bit>0] = (img8bit[img8bit>0] - img8bit_min)/(img8bit_max - img8bit_min)*255
    img8bit = img8bit.astype('uint8')

    distances = []
    max_shape = max(np.sum(np.sum(img8bit, axis=0) > 0), np.sum(np.sum(img8bit, axis=1) > 0))
    distances.append(max(int(max_shape/100), 1))
    distances.append(max(int(max_shape/20), 1))
    distances.append(max(int(max_shape/10), 1))
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    glcm = skfeat.graycomatrix(np.squeeze(img8bit), distances, angles, symmetric=True, normed=False, levels=256)
    glcm = glcm.sum(axis=3, keepdims=True)
    glcm[0,...]   = 0
    glcm[:,0,...] = 0

    glcm_norm = glcm / np.sum(glcm, axis=(0, 1), keepdims=True).astype(glcm.dtype)
    return glcm_norm

def run_glcm_features(img):
    glcm_norm = compute_glcm(img)
    homogeneity   = skfeat.graycoprops(glcm_norm, 'homogeneity')[:, 0]
    correlation   = skfeat.graycoprops(glcm_norm, 'correlation')[:, 0]
    contrast      = skfeat.graycoprops(glcm_norm, 'contrast')[:, 0]
    dissimilarity = skfeat.graycoprops(glcm_norm, 'dissimilarity')[:, 0]
    ASM           = skfeat.graycoprops(glcm_norm, 'ASM')[:, 0]
    energy        = np.sqrt(ASM)

    glcm_dict = {}
    for i in range(3):
        glcm_dict.update({
            f'homogeneity_{i+1}': homogeneity[i], 
            f'correlation_{i+1}':correlation[i],
            f'contrast_{i+1}':contrast[i],
            f'dissimilarity_{i+1}':dissimilarity[i],
            f'ASM_{i+1}':ASM[i],
            f'energy_{i+1}':energy[i],
        })

    return glcm_dict

def get_master_df(vindr_df, ddsm_df, INbreast_df):

    ddsm = ddsm_df[['left or right breast', 'image view', 'breast density', 'Manufacturer', 'full_path']].copy()
    ib = INbreast_df[['Left or Right Breast', 'Image View', 'ACR', 'Manufacturer', 'full_path']].copy()

    ddsm.columns = ['laterality', 'view_position', 'breast_density', 'Manufacturer', 'full_path']
    ib.columns = ['laterality', 'view_position', 'breast_density', 'Manufacturer', 'full_path']
    vin = vindr_df[['laterality', 'view_position', 'breast_density', 'Manufacturer', 'full_path']].copy()

    master_df = pd.concat([vin, ddsm, ib])
    return master_df

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
    return match_histograms(image,REF_IMAGE)

def denoise_image(image):
    image_8bit = np.uint8(image / 256)
    denoised_image_8bit = cv2.fastNlMeansDenoising(image_8bit, None, 10, 7, 21)
    denoised_image_16bit = np.uint16(denoised_image_8bit) * 256
    return denoised_image_16bit

def process_image(img, vendor):
    # print(vendor)
    if(vendor == 'Planmed'):
        # img = process_image_with_contour(img)
        img = 65535-img
        img = img - 55535
        img = img/3.44
        img = np.uint16(img)
        # img = apply_clahe(img)
        # img = match_intensity(img)
        # img = denoise_image(img)
        # img = adjust_intensity(img, increase=False, strength=10000)
        # img = bilateral_filter(img, sigma_color=100, sigma_space=100)
        # img = wavelet_denoising(img)

    elif(vendor == 'SIEMENS'):
        img = apply_clahe(img)
        # img = match_intensity(img)
        # img = denoise_image(img)
        # img = wavelet_denoising(img)
        # img = gamma_correction(img, gamma=1.05)
        # img = unsharp_masking(img, alpha=1.2)
        # img = bilateral_filter(img)
    
    elif(vendor == 'Siemens_INBreast'):
        img = apply_clahe(img)
        # img = match_intensity(img)
        # img = adjust_intensity(img, increase=False, strength=10000)
        # img = gamma_correction(img, gamma=0.9)
        # img = wavelet_denoising(img)
    
    elif(vendor == 'DDSM'):
        img = img/23.948
        img = np.uint16(img)
        img = apply_clahe(img)
        # img = match_intensity(img)
        # img = adjust_intensity(img, increase=True, strength=10000)
        # img = wavelet_denoising(img)
    
    elif(vendor in ['IMS s.r.l', 'IMS GIOTTO S.p.A.', 'IMS']):
        _, thresholded = cv2.threshold(img, 0, 65535, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = cv2.bitwise_and(img, img, mask=thresholded.astype(np.uint8))
        img = np.where(thresholded == 0, 0, img)
        img = img/6.601
        img = np.uint16(img)
        img = apply_clahe(img, clip_limit=8.0, tile_grid_size=(3, 3))
        # img = match_intensity(img)
        # img = wavelet_denoising(img)
        # img = adjust_intensity(img, increase=True, strength=10000)
        # img = unsharp_masking(img)
        # img = gamma_correction(img, gamma=1.05)

    return img