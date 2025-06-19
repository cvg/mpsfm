import numpy as np
import torch
from moge.model.v2 import MoGeModel

from mpsfm.extraction.base_model import BaseModel


class MoGev2(BaseModel):
    default_conf = {
        "return_types": ["depth", "normals", "valid"],
        "model_name": "Ruicheng/moge-2-vitl-normal",
        "output_coords": "bni",
        "require_download": False,
    }
    name = "mogev2"

    def _init(self, conf):

        self.model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").cuda()

        if self.conf.output_coords == "bni":
            self.output_coords = self.omni_to_bni

    def _forward(self, data):
        image = data["image"]

        image = torch.tensor(image / 255, dtype=torch.float32, device="cuda").permute(2, 0, 1)
        output = self.model.infer(image)
        image_flipped = torch.flip(image, dims=[2])
        output_flipped = self.model.infer(image_flipped)
        outdict = dict(
            depth=output["depth"],
            normals=output["normal"],
            valid=output["mask"],
            depth2=torch.flip(output_flipped["depth"], dims=[1]),
            normals2=torch.flip(output_flipped["normal"], dims=[1]),
            valid2=torch.flip(output_flipped["mask"], dims=[1]),
        )
        out_kwargs = {key: val.cpu().numpy() for key, val in outdict.items()}
        out_kwargs["normals"] = self.output_coords(out_kwargs["normals"])
        out_kwargs["normals2"] = self.output_coords(out_kwargs["normals2"])
        out_kwargs["normals2"][..., 0] *= -1
        out_kwargs["normals"][np.isinf(out_kwargs["depth"])] = np.array([0, 0, 1])
        out_kwargs["normals2"][np.isinf(out_kwargs["depth2"])] = np.array([0, 0, 1])
        out_kwargs["depth"][np.isinf(out_kwargs["depth"])] = 2
        out_kwargs["depth2"][np.isinf(out_kwargs["depth2"])] = 2
        return out_kwargs

    @staticmethod
    def omni_to_bni(normals):
        normals[..., 1:] = -normals[..., 1:]
        return normals
