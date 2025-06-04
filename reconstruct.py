import argparse
from pathlib import Path

from mpsfm.test.simple import SimpleTest
from mpsfm.utils.tools import load_cfg
from mpsfm.vars import gvars

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", type=str, default="local/example", help="Main data dir storing inputs and later the outputs"
)
parser.add_argument("--images_dir", type=str, help="Images directory")
parser.add_argument(
    "--imnames", type=str, nargs="*", help="List of image names to process. Leave empty to process all images"
)
parser.add_argument("--intrinsics_pth", type=str, default=None, help="Path to intrinsics .yaml file")
parser.add_argument("--refrec_dir", type=str, default=None, help="Path to reference reconstruction")
parser.add_argument("--cache_dir", type=str, default=None, help="Path to cache directory")
parser.add_argument("-e", "--extract", nargs="*", type=str, default=[], help="List of priors to force reextract")
parser.add_argument("-c", "--conf", type=str, help="Name of the sfm config file", default="sp-lg_m3dv2")
parser.add_argument("-v", "--verbose", type=int, default=0)
parser.add_argument(
    "--extrinsics",
    action="store_true",
    help="If set, will use extrinsics from the reference reconstruction."
    "Pipeline will only triangulate points and refine depth maps.",
)

args, _ = parser.parse_known_args()
conf = load_cfg(gvars.SFM_CONFIG_DIR / f"{args.conf}.yaml", return_name=False)
conf.extract = args.extract
conf.verbose = args.verbose
experiment = SimpleTest(conf)
mpsfm_rec = experiment(
    imnames=args.imnames,
    intrinsics_pth=args.intrinsics_pth,
    refrec_dir=args.refrec_dir,
    cache_dir=args.cache_dir,
    data_dir=args.data_dir,
    images_dir=args.images_dir,
    extrinsics=args.extrinsics,
)
sfm_outputs_dir = Path(args.data_dir) / "sfm_outputs"
sfm_outputs_dir.mkdir(exist_ok=True)
mpsfm_rec.write(sfm_outputs_dir)
