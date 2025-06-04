import traceback
from pathlib import Path

import yaml
from omegaconf import OmegaConf

from mpsfm.eval.sfm.relative_pose import EvalRelativePose
from mpsfm.sfm.reconstruction_manager import ReconstructionManager


class ArgsToConf:
    """Base class for converting command line arguments to configuration."""

    def __call__(self, args, conf, return_args=False):
        args, conf = self._args_to_conf(args, conf)
        conf.terminate = args.terminate
        conf.extract = args.extract
        conf.testset_id = args.testset_id
        conf.verbose = args.verbose
        conf.overwrite = args.overwrite
        if return_args:
            return args, conf
        return conf

    def _args_to_conf(self, args, conf):
        if args.scene is not None:
            conf.scene = args.scene
        conf.mode = args.mode
        return args, conf


class BaseTest:
    """Base class for all tests. It handles the initialization of the test
    and optionally the evaluation of the reconstruction."""

    base_default_conf = {
        "overwrite": False,
        "terminate": False,
        "reconstruction_manager": {
            "mapper": {
                "dataset": {},
            }
        },
        "eval_reconstruction": {},
        "verbose": 0,
        "registration_method": {},
        "reconstruction": {},
        "scene": None,
    }

    args_to_conf = ArgsToConf

    default_conf = {}

    sfm_outputs_dir_template = "{exp_dir}/reconstruction/{testset_type}/{scene}/{testset_desc}/{conf}"
    cache_output_dir_template = "{h5_dir}/{scene}"

    dataset = None
    parser = None

    def _init(self, *args, **kwargs):
        """Added to be modified by the child class."""

    def _auto_download(self):
        """Optional method to be implemented by the child class."""
        pass

    def auto_download(self):
        files_exist = True
        # chekck if base dir exists
        if (
            not self.dataset.data_dir.exists()
            or not all(
                [scene in set(el.name for el in self.dataset.data_dir.iterdir()) for scene in self.dataset.scenes]
            )
            or not all(
                [
                    {"images", "rec"}.issubset(set(el.name for el in scene_dir.iterdir()))
                    for scene_dir in self.dataset.data_dir.iterdir()
                    if scene_dir.is_dir()
                ]
            )
            or not all(
                any((scene_dir / "images").iterdir()) and any((scene_dir / "rec").iterdir())
                for scene_dir in self.dataset.data_dir.iterdir()
                if scene_dir.is_dir()
            )
        ):
            files_exist = False
        if not files_exist:
            self._auto_download()

    def __init__(self, conf, verbose=1):
        """Perform some logic and call the _init method of the child model."""
        default_conf = OmegaConf.merge(
            OmegaConf.create(self.base_default_conf),
            OmegaConf.create(self.default_conf),
        )
        if isinstance(conf, dict):
            conf = OmegaConf.create(conf)
        self.conf = OmegaConf.merge(default_conf, conf)
        self.eval_obj = EvalRelativePose({})
        self.conf.reconstruction_manager.verbose = self.conf.verbose
        if "extract" in self.conf:
            self.conf.reconstruction_manager.extract = self.conf.extract

        if self.dataset is None:
            raise NotImplementedError("Please set the dataset class")
        self._init()
        self.auto_download()

    def read_testsets(self, scene, mode):
        parser = self.parser(scene)
        yaml_path = self.dataset.testsets / scene / f"{mode}.yaml"
        if not yaml_path.exists():
            return None

        with open(yaml_path) as f:
            testsets = yaml.safe_load(f)

        # optionally filter by a single testset_id
        tid = self.conf.get("testset_id", None)
        if tid is not None:
            testsets = {tid: testsets[tid]}
        return {
            "testsets": testsets,
            "scene_parser": parser,
        }

    @classmethod
    def init_with_parser(cls, conf=None, parser=None, **kwargs):
        """Alternative initializer that handles the parser."""
        args_to_conf = cls.args_to_conf()
        conf = args_to_conf(parser, conf)
        instance = cls(conf, **kwargs)
        return instance

    def yield_reconstruction_info(self):
        scenes = self.dataset.scenes if self.conf.scene is None else [self.conf.scene]
        for scene in scenes:
            out_dict = self.read_testsets(scene, self.conf.mode)
            if out_dict is None:
                print(f"{out_dict} does not exist")
                continue
            testsets = out_dict["testsets"]
            for id, ref_imids in testsets.items():
                parser = out_dict["scene_parser"]

                references = [parser.rec.images[imid].name for imid in ref_imids]
                yield {
                    "scene": scene,
                    "scene_parser": parser,
                    "testset_type": self.conf.mode,
                    "ref_imids": ref_imids,
                    "references": references,
                    "testset_desc": str(id),
                }

    def experiment_exists(self, results_dir):
        """Check if the experiment results already exist. If any do not exist or are invalid, return False."""
        if not self.eval_obj.load(results_dir):
            return False
        return self.eval_obj.valid()

    def __call__(self, extract_only=False):
        models = {}
        for reconstruction_info in self.yield_reconstruction_info():
            try:
                init_info = self.init_experiment(**reconstruction_info)
                init_info["sfm_outputs_dir"].mkdir(parents=True, exist_ok=True)
                init_info["cache_dir"].mkdir(parents=True, exist_ok=True)
                results_dir = init_info["sfm_outputs_dir"] / "results"
                results_dir.mkdir(parents=True, exist_ok=True)
                if not self.conf.overwrite and self.experiment_exists(results_dir):
                    print(f"Skipping {init_info['scene']} as results already exist")
                    continue

                ReconstructionManager.freeze_conf = False
                self.reconstruction_manager = ReconstructionManager(self.conf, models=models)
                mpsfm_rec = self.reconstruction_manager(
                    extract_only=extract_only,
                    **init_info,
                )
                models = self.reconstruction_manager.models

                if extract_only:
                    continue
                self.eval_obj.setup(mpsfm_rec, init_info["scene_parser"].rec)

                print(f"Evaluating reconstruction for {init_info['scene']}...")
                self.eval_obj.compute()
                self.eval_obj.summarize()
                print(f"Saving results to {results_dir}...")
                self.eval_obj.save_results(results_dir)
            except Exception as e:
                if self.conf.terminate:
                    raise e
                traceback.print_exc()

    def init_experiment(self, **reconstruction_info):
        scene, testset_type, testset_desc = (
            reconstruction_info[key] for key in ["scene", "testset_type", "testset_desc"]
        )
        reconstruction_info["sfm_outputs_dir"] = Path(
            self.sfm_outputs_dir_template.format(
                exp_dir=self.dataset.default_exp_dir,
                testset_type=testset_type,
                scene=scene,
                testset_desc=testset_desc,
                conf=self.conf.name,
            )
        )
        reconstruction_info["cache_dir"] = Path(
            self.cache_output_dir_template.format(h5_dir=self.dataset.default_cache_dir, scene=scene)
        )
        return reconstruction_info
