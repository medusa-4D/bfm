import yaml
import torch
import numpy as np
from pathlib import Path
from .encoders import BfmEncoder
from .decoders import BFM
from .transforms import create_perspective_matrix, create_viewport_matrix, crop_matrix_to_3d


class Deep3D:

    def __init__(self, device='cuda', tform=None, img_size=None):
        """Initialize this model class."""
        
        self.device = device
        self.tform = tform
        self.img_size = img_size
        self._crop_img_size = (224, 224)
        self.cfg = self._load_cfg()
        self._setup_submodels()
        torch.set_grad_enabled(False)

    def _load_cfg(self):
        
        data_dir = Path(__file__).parent / 'data'
        cfg = data_dir / 'config.yaml'

        with open(cfg, "r") as f_in:
            cfg = yaml.safe_load(f_in)

        return cfg

    def _setup_submodels(self):        
        
        self.E_bfm = BfmEncoder()
        self.E_bfm.to(self.device)
        cp = torch.load(self.cfg['deep3d_path'])
        self.E_bfm.load_state_dict(cp['net_recon'])
        self.D_bfm = BFM(self.cfg['bfm_path'])
        self.D_bfm.to(self.device)
    
    def decompose_code(self, enc_code):
        
        id_coeffs = enc_code[:, :80]
        exp_coeffs = enc_code[:, 80: 144]
        tex_coeffs = enc_code[:, 144: 224]
        angles = enc_code[:, 224: 227]
        gammas = enc_code[:, 227: 254]
        translations = enc_code[:, 254:]
 
        return {
            'id': id_coeffs,
            'exp': exp_coeffs,
            'tex': tex_coeffs,
            'angle': angles,
            'gamma': gammas,
            'trans': translations
        }
    
    def get_faces(self):
        return self.D_bfm.face_buf.cpu().numpy()
    
    def __call__(self, img):
        
        enc_code = self.E_bfm(img)
        enc_code = self.decompose_code(enc_code)
        v, pose = self.D_bfm(enc_code)
        v = v.cpu().numpy().squeeze()
        v[:, 2] -= 10
        pose = pose.cpu().numpy().squeeze()

        PP = create_perspective_matrix(*self._crop_img_size, fx=1015.)  # forward (world -> cropped NDC)
        VP = create_viewport_matrix(*self._crop_img_size)  # forward (cropped NDC -> cropped raster)
        CP = crop_matrix_to_3d(self.tform)  # crop matrix
        VP_ = create_viewport_matrix(*self.img_size)  # backward (full NDC -> full raster)
        PP_ = create_perspective_matrix(*self.img_size, fx=1015.)  # backward (full NDC -> world)

        v = np.c_[v, np.ones(v.shape[0])]
        v = v @ PP.T
        w = v[:, 3, None].copy()
        v /= w
        v = v @ VP.T
        v = v @ np.linalg.inv(CP).T

        v = v @ np.linalg.inv(VP_).T
        v *= w
        v = v @ np.linalg.inv(PP_).T
        v = v[:, :3]
        
        return v, pose
