import cv2
import yaml
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.io import loadmat
from dlib import shape_predictor, get_frontal_face_detector

from .align import align_img


class BfmCropModel:
    
    def __init__(self, device='cuda'):
        self.device = device
        self.lm_ref = self._load_data()
        self._detector = get_frontal_face_detector()
        self._lm_predictor = self._init_lm_predictor()

    def _load_data(self):

        data_dir = Path(__file__).parents[1] / 'data/similarity_Lm3D_all.mat'
        lm_ref = loadmat(data_dir)['lm']
        
        # calculate 5 facial landmarks using 68 landmarks
        lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
        lm_ref = np.stack([lm_ref[lm_idx[0], :],
                           np.mean(lm_ref[lm_idx[[1, 2]], :], 0),
                           np.mean(lm_ref[lm_idx[[3, 4]], :], 0),
                           lm_ref[lm_idx[5], :], lm_ref[lm_idx[6], :]], axis=0)
        lm_ref = lm_ref[[1, 2, 0, 3, 4], :]

        return lm_ref

    def _init_lm_predictor(self):
        
        data_dir = Path(__file__).parents[1] / 'data'
        cfg = data_dir / 'config.yaml'

        with open(cfg, "r") as f_in:
            cfg = yaml.safe_load(f_in)
        
        return shape_predictor(cfg['dlib_path'])

    def _detect_landmarks(self, img):
        
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        rects = self._detector(gray, 1)
        shape = self._lm_predictor(gray, rects[0])
    
    	# Convert it to the NumPy Array
        lm68 = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            lm68[i] = (shape.part(i).x, shape.part(i).y)
        
        return lm68

    def __call__(self, img_path):

        img = Image.open(img_path)
        W, H = img.size
        lm68 = self._detect_landmarks(img)
        lm68 = lm68.reshape([-1, 2])
        lm68[:, -1] = H - 1 - lm68[:, -1]
        img_crop, tform = align_img(img, lm68, self.lm_ref)
        self.tform = tform
        img = img_crop / 255.
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        
        return img.to(self.device)