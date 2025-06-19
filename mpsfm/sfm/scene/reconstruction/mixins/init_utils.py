from copy import deepcopy
from pathlib import Path

import pycolmap
from tqdm import tqdm

from mpsfm.sfm.scene.camera import Camera
from mpsfm.sfm.scene.image import Image
from mpsfm.utils.tools import load_cfg


class InitUtils:
    @classmethod
    def initialize_from_reconstruction(cls, conf, scene_parser, references: list[str] = None):
        """Initializes reconstruction"""
        inst = cls(conf)
        inst.scene_parser = scene_parser
        inst.references = references
        if scene_parser.reconstruction_dir is not None:
            refrec = pycolmap.Reconstruction(scene_parser.reconstruction_dir)
        else:
            refrec = scene_parser.rec
        add_cameras = {camera.camera_id: camera for camera in refrec.cameras.values()}
        for camera in add_cameras.values():
            inst.add_camera(deepcopy(camera))
        for imid, image in refrec.images.items():
            if references is not None and Path(image.name).name not in references:
                continue
            image_ = pycolmap.Image(image_id=imid, name=Path(image.name).name, camera_id=image.camera_id)
            inst.add_image(image_)
        inst._images = {
            imid: Image(inst.conf.image, image=image)
            for imid, image in inst.rec.images.items()
            if imid in refrec.images
        }
        camids = {image.camera_id for image in inst.rec.images.values()}
        inst._cameras = {camid: Camera(inst.rec.cameras[camid]) for camid in camids}
        return inst

    def initialize_mono_maps(self, extraction_obj=None, **kwargs):
        print("Initializing camera integration data...")
        cam_to_imid = {image.camera_id: imid for imid, image in self.images.items()}
        for cam_id, camera in self.cameras.items():
            imid = cam_to_imid.get(cam_id)
            resolution = self.conf.normscale / max(camera.width, camera.height)
            H = round(camera.height * resolution)
            W = round(camera.width * resolution)

            self.cameras[cam_id].init_int_data(H, W)
            self.cameras[cam_id].sx = resolution
            self.cameras[cam_id].sy = resolution

        for imid in tqdm(self.images):
            cam_id = self.images[imid].camera_id
            self.images[imid].init_depth(
                camera=self.cameras[cam_id],
                imid=imid,
                mpsfm_rec=self,
                kps=self.keypoints(imid),
                depth_dir=extraction_obj.depth_dir,
                normals_dir=extraction_obj.normals_dir,
                masks_path=[extraction_obj.masks_dirs[mask] for mask in extraction_obj.masks],
                **kwargs,
            )
        return True

    def init_kps_info(self, extraction_obj):

        if self.conf.matches_mode == "sparse":
            from mpsfm.extraction.imagewise.features import CONFIG_DIR

            kps_conf = load_cfg(CONFIG_DIR / f"{extraction_obj.conf.features}.yaml")
            std_mult = 1
            resize = (
                kps_conf["preprocessing"]["resize_max"]
                if kps_conf["preprocessing"].get("resize_force", False)
                else None
            )
        else:
            from mpsfm.extraction.pairwise import CONFIG_DIR

            std_mult = 0.5  # we expect keypoint to be more repeatable than with sparse features
            kps_conf = load_cfg(CONFIG_DIR / f"{extraction_obj.conf.matcher}.yaml")
            resize = kps_conf.resize

        for imid, image in self.images.items():
            H, W = image.camera.height, image.camera.width
            resized_by = resize / max(H, W) if resize is not None else 1
            self.images[imid].kp_std = max(
                1, std_mult / resized_by
            )  # don't expect keypoint to be more precise than original resolution
            if self.conf.verbose > 1:
                print(f"Image resize from {(H, W)} to {(H * resized_by, W * resized_by)} for {image.name}")
                print(f"Setting kp_std for {imid} to {self.images[imid].kp_std}")
