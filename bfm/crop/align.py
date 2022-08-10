"""This script contains the image preprocessing code for Deep3DFaceRecon_pytorch
"""
import cv2
import warnings
import numpy as np
from PIL import Image
from skimage.transform import warp, SimilarityTransform, estimate_transform

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 


# calculating least square problem for image alignment
def POS(xp, x):
    npts = xp.shape[1]

    A = np.zeros([2*npts, 8])

    A[0:2*npts-1:2, 0:3] = x.transpose()
    A[0:2*npts-1:2, 3] = 1

    A[1:2*npts:2, 4:7] = x.transpose()
    A[1:2*npts:2, 7] = 1

    b = np.reshape(xp.transpose(), [2*npts, 1])

    k, _, _, _ = np.linalg.lstsq(A, b)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2
    t = np.stack([sTx, sTy], axis=0)

    return t, s


# utils for landmark detection
def img_padding(img, box):
    success = True
    bbox = box.copy()
    res = np.zeros([2*img.shape[0], 2*img.shape[1], 3])
    res[img.shape[0] // 2: img.shape[0] + img.shape[0] //
        2, img.shape[1] // 2: img.shape[1] + img.shape[1]//2] = img

    bbox[0] = bbox[0] + img.shape[1] // 2
    bbox[1] = bbox[1] + img.shape[0] // 2
    if bbox[0] < 0 or bbox[1] < 0:
        success = False
    return res, bbox, success

# utils for landmark detection
def crop(img, bbox):
    padded_img, padded_bbox, flag = img_padding(img, bbox)
    if flag:
        crop_img = padded_img[padded_bbox[1]: padded_bbox[1] +
                            padded_bbox[3], padded_bbox[0]: padded_bbox[0] + padded_bbox[2]]
        crop_img = cv2.resize(crop_img.astype(np.uint8),
                            (224, 224), interpolation=cv2.INTER_CUBIC)
        scale = 224 / padded_bbox[3]
        return crop_img, scale
    else:
        return padded_img, 0


# resize and crop images for face reconstruction
def resize_n_crop_img(img, lm, t, s, target_size=224.):
    w0, h0 = img.size
    w = (w0*s).astype(np.int32)
    h = (h0*s).astype(np.int32)
    left = (w/2 - target_size/2 + float((t[0] - w0/2)*s)).astype(np.int32)
    right = left + target_size
    up = (h/2 - target_size/2 + float((h0/2 - t[1])*s)).astype(np.int32)
    below = up + target_size

    # Added by Lukas, based on a helpful comment here:
    # https://github.com/sicxu/Deep3DFaceRecon_pytorch/issues/87
    scale_tform = SimilarityTransform(scale=s)
    dst = np.array([[0, 0], [0, target_size - 1], [target_size - 1, 0]])
    src = np.array([[left, up], [left, below], [right, up]])
    trans_tform = estimate_transform('similarity', src, dst)
    tform = scale_tform + trans_tform
    img_crop = warp(np.array(img), tform.inverse, output_shape=(224, 224), preserve_range=True)
    img_crop = img_crop.astype(np.uint8)
    #cv2.imwrite('testtest.jpg', img_crop)

    #img = img.resize((w, h), resample=Image.BICUBIC)
    #img = img.crop((left, up, right, below))

    #lm = np.stack([lm[:, 0] - t[0] + w0/2, lm[:, 1] -
    #              t[1] + h0/2], axis=1)*s
    #lm = lm - np.reshape(
    #        np.array([(w/2 - target_size/2), (h/2-target_size/2)]), [1, 2])

    return img_crop, tform


def extract_5p(lm):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lm5p = np.stack([lm[lm_idx[0], :], np.mean(lm[lm_idx[[1, 2]], :], 0), np.mean(
        lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]], axis=0)
    lm5p = lm5p[[1, 2, 0, 3, 4], :]
    return lm5p


# utils for face reconstruction
def align_img(img, lm, lm3D, target_size=224., rescale_factor=102.):
    """
    Return:
        transparams        --numpy.array  (raw_W, raw_H, scale, tx, ty)
        img_new            --PIL.Image  (target_size, target_size, 3)
        lm_new             --numpy.array  (68, 2), y direction is opposite to v direction
        mask_new           --PIL.Image  (target_size, target_size)
    
    Parameters:
        img                --PIL.Image  (raw_H, raw_W, 3)
        lm                 --numpy.array  (68, 2), y direction is opposite to v direction
        lm3D               --numpy.array  (5, 3)
        mask               --PIL.Image  (raw_H, raw_W, 3)
    """

    lm5p = extract_5p(lm)
    
    # calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
    t, s = POS(lm5p.transpose(), lm3D.transpose())
    t = t.squeeze()
    s = rescale_factor/s

    # processing the image
    img_new, tform = resize_n_crop_img(img, lm5p, t, s, target_size=target_size)

    return img_new, tform
