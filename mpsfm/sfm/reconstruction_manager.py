from mpsfm.baseclass import BaseClass
from mpsfm.sfm.mapper import MpsfmMapper, MpsfmRefiner

class ReconstructionManager(BaseClass):
    """Used to create the reconstruction object and manage the reconstruction process."""

    freeze_conf = False

    def _init(self, models=None):
        if models is None:
            models = {}
        self.models = models
        self.mapper = None

    def __call__(
        self,
        references,
        cache_dir,
        sfm_outputs_dir,
        scene_parser,
        scene="<custom>",
        extract_only=False,
        extrinsics=False,
        **kwargs,
    ):

        exclude_init_pairs = set()
        print(50 * "=")
        if extract_only:
            print("\tSTARTING EXTRACTION")
            self.log(f"for {scene} and images {references} with imids {kwargs['ref_imids']}", level=1)
        else:
            print("\tSTARTING RECONSTRUCTION")
            self.log(f"for {scene} and images {references} with imids {kwargs['ref_imids']}", level=1)
        print(50 * "=")
        if extrinsics:
            mapper_class = MpsfmRefiner
        else:
            mapper_class = MpsfmMapper
        mapper_class.freeze_conf = False
        self.mapper = mapper_class(
            conf=self.conf,
            references=references,
            cache_dir=cache_dir,
            sfm_outputs_dir=sfm_outputs_dir,
            scene=scene,
            scene_parser=scene_parser,
            models=self.models,
            extract_only=extract_only,
            **kwargs,
        )
        # check if has atribute extractor
        if hasattr(self.mapper, "extractor"):
            self.models = self.mapper.extractor.models
        elif hasattr(self.mapper, "models"):
            self.models = self.mapper.models

        if extract_only:
            print("Extraction complete")
            return None
        mpsfm_rec, _ = self.mapper(
            refrec=scene_parser.rec, exclude_init_pairs=exclude_init_pairs, references=references
        )
        print(
            f"\nReconstrtuction complete with ({mpsfm_rec.num_reg_images()}/"
            f"{mpsfm_rec.num_images()}) registered images"
        )
        print(f"Rec has {mpsfm_rec.num_reg_images()}/{mpsfm_rec.num_images()} registered images")
        if self.conf.verbose:
            self.mapper.visualization()
        return mpsfm_rec
