from pathlib import Path

from mpsfm.baseclass import BaseClass
from mpsfm.data_proc.simple import SimpleDataset, SimpleParser
from mpsfm.sfm.reconstruction_manager import ReconstructionManager


class SimpleTest(BaseClass):
    """SimpleTest class for reconstructing a user input testset."""

    dataset = SimpleDataset
    default_conf = {"dataset": {"name": "simple"}}
    freeze_conf = False

    def __call__(
        self,
        imnames=None,
        data_dir=None,
        intrinsics_pth=None,
        refrec=None,
        refrec_dir=None,
        cache_dir=None,
        images_dir=None,
    ):
        data_dir = Path(data_dir)
        cache_dir = Path(cache_dir) if cache_dir is not None else None
        images_dir = Path(images_dir) if images_dir is not None else None

        scene_parser = SimpleParser(
            data_dir=data_dir,
            imnames=imnames,
            intrinsics_pth=intrinsics_pth,
            refrec=refrec,
            refrec_dir=refrec_dir,
            rgb_dir=images_dir,
        )

        if cache_dir is None:
            cache_dir = data_dir / "cache_dir"

        self.reconstruction_manager = ReconstructionManager(self.conf)
        init_info = dict(
            references=scene_parser.imnames,
            sfm_outputs_dir=Path(data_dir) / "sfm_outputs",
            cache_dir=cache_dir,
            ref_imids=list(scene_parser.rec.images.keys()),
            scene_parser=scene_parser,
        )
        return self.reconstruction_manager(**init_info)
