import numpy as np

from mpsfm.baseclass import BaseClass
from mpsfm.sfm.scene.image.depth import Depth
from mpsfm.sfm.scene.image.integration import Integration
from mpsfm.sfm.scene.image.normals import Normals
from mpsfm.utils.io import get_mask, get_mono_map, get_mono_map_from_pairs


class ColmapImageWrapper:
    def __getattr__(self, name):
        if name == "_image" or "_image" not in self.__dict__:
            raise AttributeError(f"{name} not found and _image not initialized")
        if name == "camera":
            return self._camera
        return getattr(self._image, name)

    def __setattr__(self, name, value):
        if name == "_image" or "_image" not in self.__dict__:
            super().__setattr__(name, value)
        elif hasattr(self._image, name):
            setattr(self._image, name, value)
        else:
            super().__setattr__(name, value)


class Image(ColmapImageWrapper, BaseClass, Integration):
    """MP-SfM image wrapper for COLMAP image."""

    default_conf = {
        "depth": {},
        "normals": {},
        "depth_type": "<--->",
        "verbose": 0,
        # integration
        "large_number": 1e6,
        "max_iter": 10,
        "tol": 5e-2,
        "step_size": 1,
        "cg_max_iter": 5000,
        "cg_tol": 1e-3,
        "lambda1": 1,
        "lambda2": 1,
        "k": 1,
        "depth_magnitude_multiplier": 1,
        "normals_magnitude_multiplier": 1,
        "cov_ignore_depth": True,
        "downscale_factor": 2,
        "downscaled": True,
        "scale_filter": True,
        "scale_filter_factor": 1.5,
        # dev
        "robust_triangles": 2,
        "ignore_depths": True,
    }

    def _propagate_conf(self):
        self.conf.depth.verbose = self.conf.verbose
        self.conf.normals.verbose = self.conf.verbose

    def _init(self, image):
        """Initialize and integrate Depth and Normals."""
        self._image = image

        try:
            self._camera = image.camera
        except AttributeError:
            print("Warning: not loading camera. Fine if for benchmarking old colmap version.")

        Integration.__init__(self)

        self.ignore_matches_AP = {}
        self.dc_times_inliers_resampled = 0
        self.last_dc_score = None
        self.failed_dc_check = False
        self.failed_normal_registration = False

        self.depth_dir = None
        self.normals_dir = None
        self.imid = None
        self.mpsfm_rec = None
        self.image = None
        self.masks_path = None
        self.depth = None
        self.normals = None

    def init_depth(self, camera, depth_dir, normals_dir, imid, mpsfm_rec, masks_path, **kwargs):
        """Initialize Depth and Normals."""
        self._camera = camera
        self.depth_dir = depth_dir
        self.normals_dir = normals_dir
        self.imid = imid
        self.mpsfm_rec = mpsfm_rec
        self.image = self.mpsfm_rec.images[self.imid]
        self.masks_path = masks_path

        mask = self.load_masks_data()
        if self.conf.depth_type == "monocular":
            depth_dict = self.load_depth_data(**kwargs)
            if depth_dict is None:
                return
            self.depth = Depth(self.conf.depth, depth_dict=depth_dict, camera=camera, mask=mask, **kwargs)
        else:
            raise ValueError(f"Unknown depth type: {self.conf.depth_type}")
        normals_dict = self.load_normals_data()
        self.normals = Normals(
            self.conf.normals,
            normals_dict=normals_dict,
            camera=camera,
            continuity_mask=self.depth.continuity_mask,
            mask=mask,
            **kwargs,
        )

    def load_depth_data(self, pairs_pth=None, **kwargs):
        """Load depth data from depth directory."""
        if "mast3r" in self.depth_dir.name:
            depth_dict = get_mono_map_from_pairs(self.depth_dir, self.name, pairs_pth)
        else:
            depth_dict = get_mono_map(self.depth_dir, self.name)
        return depth_dict

    def load_normals_data(self):
        """Load normals data from normals directory."""
        normals_dict = get_mono_map(self.normals_dir, self.name)
        return normals_dict

    def load_masks_data(self, masks_path=None):
        """Load mask data from mask directory."""
        if masks_path is None:
            masks_path = self.masks_path
        masks = []
        for mask_path in masks_path:
            masks.append(get_mask(mask_path, self.name))
        mask = np.prod(masks, axis=0) if len(masks) > 0 else None
        return mask

    def keypoint_coords_with_3d(self):
        pts2d_indices_with_3d = np.where(np.array(self.point3D_ids()) != 18446744073709551615)[0]
        return self.keypoint_coordinates(pts2d_indices_with_3d)
