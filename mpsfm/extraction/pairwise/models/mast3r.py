import os
import sys
from pathlib import Path

import numpy as np
import PIL.Image
import torch
import torchvision.transforms as tvf
from PIL.ImageOps import exif_transpose

from mpsfm.extraction import device
from mpsfm.extraction.base_model import BaseModel
from mpsfm.vars import gvars

from .utils.featuremap import NNs_sparse

try:
    from pillow_heif import register_heif_opener  # noqa

    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False

mast3r_root_dir = gvars.ROOT / "third_party/mast3r"
sys.path.append(str(mast3r_root_dir))  # noqa: E402
dust3r_root_dir = gvars.ROOT / "third_party/mast3r/dust3r"
sys.path.append(str(dust3r_root_dir))  # noqa: E402
curope_root_dir = gvars.ROOT / "third_party/mast3r/dust3r/croco/models/curope"
sys.path.append(str(curope_root_dir))  # noqa: E402

from dust3r.utils.image import ImgNorm, _resize_pil_image  # noqa: E402
from mast3r.fast_nn import fast_reciprocal_NNs  # noqa: E402
from mast3r.model import AsymmetricMASt3R, load_model  # noqa: E402


def symmetric_inference(model, img1, img2):
    shape1 = torch.tensor(img1.shape[-2:])[None].to(device, non_blocking=True)
    shape2 = torch.tensor(img2.shape[-2:])[None].to(device, non_blocking=True)
    img1 = img1.to(device, non_blocking=True)
    img2 = img2.to(device, non_blocking=True)

    # compute encoder only once
    feat1, feat2, pos1, pos2 = model._encode_image_pairs(img1, img2, shape1, shape2)

    def decoder(feat1, feat2, pos1, pos2, shape1, shape2):
        dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            res1 = model._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = model._downstream_head(2, [tok.float() for tok in dec2], shape2)
        return res1, res2

    # decoder 1-2
    res11, res21 = decoder(feat1, feat2, pos1, pos2, shape1, shape2)
    # decoder 2-1
    res22, res12 = decoder(feat2, feat1, pos2, pos1, shape2, shape1)

    return (res11, res21, res22, res12)


def load_images(folder_or_list, size, square_ok=False, verbose=True):
    """open and convert all images in a list or folder to proper input format for DUSt3R"""
    if isinstance(folder_or_list, str):
        if verbose:
            print(f">> Loading images from {folder_or_list}")
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f">> Loading a list of {len(folder_or_list)} images")
        root, folder_content = "", folder_or_list

    else:
        raise ValueError(f"bad {folder_or_list=} ({type(folder_or_list)})")

    supported_images_extensions = [".jpg", ".jpeg", ".png"]
    if heif_support_enabled:
        supported_images_extensions += [".heic", ".heif"]
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    vars = []
    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue
        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert("RGB")
        W1, H1 = img.size
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W // 2, H // 2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx - half, cy - half, cx + half, cy + half))
        else:
            halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
            if not (square_ok) and W == H:
                halfh = int(3 * halfw / 4)
            img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

        W2, H2 = img.size
        if verbose:
            print(f" - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}")
        imgs.append(
            dict(img=ImgNorm(img)[None], true_shape=np.int32([img.size[::-1]]), idx=len(imgs), instance=str(len(imgs)))
        )
        vars.append((cx, cy, halfw, halfh, H1, W1, H2, W2, H, W))

    assert imgs, "no images foud at " + root
    if verbose:
        print(f" (Found {len(imgs)} images)")
    return imgs, vars


def merge_corres(idx1, idx2, shape1=None, shape2=None, ret_xy=True, ret_index=False):
    assert idx1.dtype == idx2.dtype == np.int32

    # unique and sort along idx1
    corres = np.unique(np.c_[idx2, idx1].view(np.int64), return_index=ret_index)

    if ret_index:
        corres, indices = corres
    xy2, xy1 = corres[:, None].view(np.int32).T

    if ret_xy:
        assert shape1 and shape2
        xy1 = np.unravel_index(xy1, shape1)
        xy2 = np.unravel_index(xy2, shape2)
        if ret_xy != "y_x":
            xy1 = xy1[0].base[:, ::-1]
            xy2 = xy2[0].base[:, ::-1]
    if ret_index:
        return xy1, xy2, indices  # [xy1_indices][xy2_indices]
    return xy1, xy2


def extract_correspondences(feats, qonfs, subsample=8, ptmap_key="pred_desc"):
    feat11, feat21, feat22, feat12 = feats
    qonf11, qonf21, qonf22, qonf12 = qonfs
    assert feat11.shape[:2] == feat12.shape[:2] == qonf11.shape == qonf12.shape
    assert feat21.shape[:2] == feat22.shape[:2] == qonf21.shape == qonf22.shape

    opt = dict(device="cpu", workers=32) if "3d" in ptmap_key else dict(device=device, dist="dot", block_size=2**13)

    # matching the two pairs
    idx1 = []
    idx2 = []
    qonf1 = []
    qonf2 = []
    for A, B, QA, QB in [(feat11, feat21, qonf11.cpu(), qonf21.cpu()), (feat12, feat22, qonf12.cpu(), qonf22.cpu())]:
        nn1to2 = fast_reciprocal_NNs(A, B, subsample_or_initxy1=subsample, ret_xy=False, **opt)
        nn2to1 = fast_reciprocal_NNs(B, A, subsample_or_initxy1=subsample, ret_xy=False, **opt)

        idx1.append(np.r_[nn1to2[0], nn2to1[1]])
        idx2.append(np.r_[nn1to2[1], nn2to1[0]])
        qonf1.append(QA.ravel()[idx1[-1]])
        qonf2.append(QB.ravel()[idx2[-1]])

    # merge corres from opposite pairs
    H1, W1 = feat11.shape[:2]
    H2, W2 = feat22.shape[:2]
    cat = np.concatenate

    xy1, xy2, idx = merge_corres(cat(idx1), cat(idx2), (H1, W1), (H2, W2), ret_xy=True, ret_index=True)
    corres = (xy1.copy(), xy2.copy(), np.sqrt(cat(qonf1)[idx] * cat(qonf2)[idx]))
    return corres


def extract_correspondences_sparse(feats, qonfs, kps0, kps1, subsample=8, scores_thresh=None, ptmap_key="pred_desc"):
    feat11, feat21, feat22, feat12 = feats
    qonf11, qonf21, qonf22, qonf12 = qonfs
    assert feat11.shape[:2] == feat12.shape[:2] == qonf11.shape == qonf12.shape
    assert feat21.shape[:2] == feat22.shape[:2] == qonf21.shape == qonf22.shape
    opt = dict(workers=32) if "3d" in ptmap_key else dict(dist="dot", block_size=2**13)
    matches = []
    scores = []

    for A, B, QA, QB in [(feat11, feat21, qonf11, qonf21), (feat12, feat22, qonf12, qonf22)]:
        matches12, scores12 = NNs_sparse(
            A, B, QA, QB, kps0, kps1, subsample_or_initxy1=subsample, ret_xy=False, scores_thresh=scores_thresh, **opt
        )
        matches.append(matches12)
        scores.append(scores12)
        break

    return matches, scores


def map_keypoints_to_original_after_crop(keypoints_crop, cx, cy, halfw, halfh):
    keypoints_scaled = keypoints_crop + np.array([cx - halfw, cy - halfh])
    return keypoints_scaled


# monkey patch for silence
class AsymmetricMASt3R(AsymmetricMASt3R):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            # added verbose
            return load_model(pretrained_model_name_or_path, device="cpu", verbose=False)
        else:
            return super().from_pretrained(pretrained_model_name_or_path, **kw)


class Mast3rMatcher(BaseModel):
    default_conf = {
        "model_name": "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
        "long_edge_size": 512,
        "window": 8,
        "nms_radius": 6,
        "NN_scores_thresh": 0.85,
        "download_url": "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
        "require_download": True,
        "download_method": "wget",
    }
    required_inputs = ["image0", "image1", "name0", "name1"]

    # Initialize the line matcher
    def _init(self, conf):
        self.net = AsymmetricMASt3R.from_pretrained(
            Path(self.conf.models_dir, self.conf["model_name"]), verbose=False
        ).to(device)

    def process_image(self, image, square_ok=False):
        H, W = image.shape[-2:]
        cx, cy = W // 2, H // 2
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if not (square_ok) and W == H:
            halfh = int(3 * halfw / 4)
        image = image[..., cy - halfh : cy + halfh, cx - halfw : cx + halfw]
        H2, W2 = image.shape[-2:]
        ImgNorm = tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image = ImgNorm(image)
        return image, (cx, cy, halfw, halfh, None, None, H2, W2, H, W)

    def _forward(self, data, mode="sparse", **kwargs):
        im0, var0 = self.process_image(data["image0"])
        im1, var1 = self.process_image(data["image1"])
        H1, W1 = var0[-2], var0[-1]
        H2, W2 = var1[-2], var1[-1]

        res = symmetric_inference(self.net, im0, im1)
        X11, _, X22, _ = [r["pts3d"][0].cpu().numpy() for r in res]
        C11, _, C22, _ = [r["conf"][0].cpu().numpy() for r in res]
        descs = [r["desc"][0] for r in res]
        qonfs = [r["desc_conf"][0] for r in res]
        pred = {}

        if "dense" in mode:
            # extracting 2v corres
            corres = extract_correspondences(descs, qonfs, subsample=self.conf.window)
            dkps0, dkps1, scores0 = corres

            dkps0 = map_keypoints_to_original_after_crop(dkps0, *var0[:-6])
            dkps1 = map_keypoints_to_original_after_crop(dkps1, *var1[:-6])

        def extract_rescale_crop(v):
            cropx_a, cropx_b = v[0] - v[2], v[0] + v[2]
            cropy_a, cropy_b = v[1] - v[3], v[1] + v[3]
            return (cropx_a, cropx_b, cropy_a, cropy_b)

        rescale_crop = [extract_rescale_crop(v) for v in [var0, var1]]
        (cropx1a, cropx1b, cropy1a, cropy1b), (cropx2a, cropx2b, cropy2a, cropy2b) = rescale_crop

        if "sparse" in mode:
            skpts0 = data["skpts0"] / data["scale0"]
            skpts1 = data["skpts1"] / data["scale1"]

            smask0 = (
                (cropx1a <= skpts0[:, 0])
                & (skpts0[:, 0] < cropx1b)
                & (cropy1a <= skpts0[:, 1])
                & (skpts0[:, 1] < cropy1b)
            )
            smask1 = (
                (cropx2a <= skpts1[:, 0])
                & (skpts1[:, 0] < cropx2b)
                & (cropy2a <= skpts1[:, 1])
                & (skpts1[:, 1] < cropy2b)
            )

            skpts0 = skpts0[smask0] - np.array([cropx1a, cropy1a])
            skpts1 = skpts1[smask1] - np.array([cropx2a, cropy2a])

            smatches, sscores = [
                el[0]
                for el in extract_correspondences_sparse(
                    descs, qonfs, skpts0, skpts1, subsample=self.conf.window, scores_thresh=self.conf.NN_scores_thresh
                )
            ]

        if "dense" in mode:
            pred["dkeypoints0"], pred["dkeypoints1"] = dkps0, dkps1
            pred["dscores"] = scores0
        if "sparse" in mode:
            pred["smatches0"] = smatches
            pred["smatching_scores0"] = sscores

        if "depth" in mode:
            for i, (H, W, slicex, slicey, X, C) in enumerate(
                [
                    (H1, W1, slice(cropx1a, cropx1b), slice(cropy1a, cropy1b), X11, C11),
                    (H2, W2, slice(cropx2a, cropx2b), slice(cropy2a, cropy2b), X22, C22),
                ]
            ):
                pred[f"depth{i}"] = np.zeros((H, W))
                pred[f"variance{i}"] = np.ones((H, W)) * 1e6
                pred[f"valid{i}"] = np.zeros((H, W), dtype=bool)
                pred[f"depth{i}"][slicey, slicex] = X[..., -1]
                pred[f"variance{i}"][slicey, slicex] = (1 / C) ** 2
                pred[f"valid{i}"][slicey, slicex] = True
        return pred
