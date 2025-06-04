"""Dataset and parser for reconstructing user provided images."""

from pathlib import Path

import pycolmap
import torch
import yaml
from PIL import Image
from pycolmap import Reconstruction

from .basedataset import BaseDataset, BaseDatasetParser


class SimpleDataset(BaseDataset, torch.utils.data.Dataset):
    """Dataset class for user provided images following standard pipeline structure."""

    def _init(self, image_list=None, scene_parser=None, images_dir=None, **kwargs):
        assert self.conf.batch_size == 1, "Batch size must be 1 for this dataset"
        if scene_parser is None:
            assert images_dir is not None, "images_dir must be provided if scene_parser is None"
            self.scene_parser = SimpleParser(rgb_dir=images_dir.parent, imnames=image_list)
        else:
            self.scene_parser = scene_parser
        if image_list is None:
            image_ids = list(self.scene_parser.rec.images.keys())
        else:
            image_ids = [k for k, v in self.scene_parser.rec.images.items() if Path(v.name).name in image_list]

        image_ids.sort()
        self.imids = image_ids


class SimpleParser(BaseDatasetParser):
    """Parser for user provided images following standard pipeline structure."""

    scene = "<custom>"
    dataset = SimpleDataset

    def __init__(self, *args, data_dir=None, imnames=None, intrinsics_pth=None, rgb_dir=None, refrec_dir=None, **kwargs):
        if rgb_dir is None:
            rgb_dir = Path(data_dir) / "images"
        if imnames is None:
            imnames = [im.name for im in rgb_dir.iterdir()]
        self.imnames = imnames
        
        if refrec_dir is not None:
            self.rec = pycolmap.Reconstruction(refrec_dir)
        else:
            self.rec = Reconstruction()
            
            if intrinsics_pth is None:
                intrinsics_pth = Path(data_dir) / "intrinsics.yaml"
                with open(intrinsics_pth, encoding="utf-8") as f:
                    intrinsics = yaml.safe_load(f)
            if len(intrinsics) == 1:
                assert intrinsics[1]["images"] == "all" or (
                    isinstance(intrinsics[1]["images"], list) and len(intrinsics[1]["images"]) == len(self.imnames)
                ), "If only one camera is provided, images must be 'all' or a list of all image names"

            image_id = 1
            for camid, camdict in intrinsics.items():
                params = camdict["params"]
                images = camdict["images"]
                if images == "all":
                    images = self.imnames

                shapes = []
                for imname in images:
                    im = Image.open(rgb_dir / imname)
                    shapes.append(im.size)
                assert len(set(shapes)) == 1, "All images must have the same shape"
                width, height = shapes[0]

                camera = pycolmap.Camera(
                    model="PINHOLE",
                    width=width,
                    height=height,
                    params=params,
                    has_prior_focal_length=True,
                    camera_id=camid,
                )
                self.rec.add_camera(camera)

                for imanme in images:
                    self.rec.add_image(pycolmap.Image(name=imanme, camera_id=camid, image_id=image_id))
                    image_id += 1
        self.reconstruction_dir = refrec_dir
        self.rgb_dir = rgb_dir
