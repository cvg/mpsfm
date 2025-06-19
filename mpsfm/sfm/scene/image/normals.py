"""Module for normals class"""

import cv2
import numpy as np

from mpsfm.baseclass import BaseClass
from mpsfm.sfm.scene.image.mixins.priorutils import PriorUtils

large_number = 1e6


def diff_angle(angle1, angle2):
    """Compute the difference between two angles in radians."""
    return np.minimum(np.abs(angle1 - angle2), 2 * np.pi - np.abs(angle1 - angle2))


def cart_to_spherical(norm):
    """Convert Cartesian coordinates to spherical coordinates."""
    norm = norm / np.linalg.norm(norm, axis=-1, keepdims=True)
    norm_spherical = np.concatenate(
        [
            np.arccos(norm[..., 2, None]),
            (
                np.sign(norm[..., 1])
                * np.arccos(norm[..., 0] / (1e-6 + (norm[..., 0] ** 2 + norm[..., 1] ** 2) ** 0.5))
            )[..., None],
        ],
        axis=-1,
    )
    return norm_spherical


def cart_mean_to_spherical(norm1, norm2):
    """Compute the mean of two normals in spherical coordinates."""
    norm1 = norm1 / np.linalg.norm(norm1, axis=-1, keepdims=True)
    norm2 = norm2 / np.linalg.norm(norm2, axis=-1, keepdims=True)
    mean_normal = (norm1 + norm2) / 2
    norm_sphere_mean = cart_to_spherical(mean_normal)
    return norm_sphere_mean, mean_normal


def cart_to_spherical_mean(norm1, norm2):
    """Compute the mean of two normals in spherical coordinates."""
    norm1_spherical = cart_to_spherical(norm1)
    norm2_spherical = cart_to_spherical(norm2)
    # make sure that the angles are closest to eachother so +2pi if needed
    diff = norm2_spherical - norm1_spherical
    sub_mask = diff > np.pi
    norm2_spherical[sub_mask] -= 2 * np.pi
    add_mask = diff < -np.pi
    norm2_spherical[add_mask] += 2 * np.pi
    norm_sphere_mean = (norm1_spherical + norm2_spherical) / 2
    return norm_sphere_mean, norm1_spherical, norm2_spherical


def covar_from_spherical(spherical_mean, sphere1, sphere2):
    """Compute the spherical covariance matrix."""
    cov_diag = ((diff_angle(sphere1, spherical_mean)) ** 2 + (diff_angle(sphere2, spherical_mean)) ** 2).clip(0)
    cov_off_diag = (diff_angle(sphere1[..., 0], spherical_mean[..., 0])) * (
        diff_angle(sphere1[..., 1], spherical_mean[..., 1])
    ) + (diff_angle(sphere2[..., 0], spherical_mean[..., 0])) * (diff_angle(sphere2[..., 1], spherical_mean[..., 1]))
    Cov_sphere = np.stack([cov_diag[..., 0], cov_off_diag, cov_off_diag, cov_diag[..., 1]], axis=-1).reshape(
        sphere1.shape[0], sphere1.shape[1], 2, 2
    )
    return Cov_sphere


def covar_sphere_thorough_spherical_mean(norm1, norm2):
    """Compute spherical covariance using the mean of the normals in spherical coordinates."""
    norm_sphere_mean, sphere1, sphere2 = cart_to_spherical_mean(norm1, norm2)

    norm1 = norm1 / np.linalg.norm(norm1, axis=-1, keepdims=True)
    norm2 = norm2 / np.linalg.norm(norm2, axis=-1, keepdims=True)
    mean_normal = (norm1 + norm2) / 2
    return (
        covar_from_spherical(norm_sphere_mean, sphere1, sphere2),
        norm_sphere_mean,
        mean_normal / np.linalg.norm(mean_normal, axis=-1, keepdims=True),
    )


def Jacobian(sphere_mean):
    """Compute the Jacobian matrix spherical to Cartesian."""
    Jcostheta = np.cos(sphere_mean[..., 0])
    Jcosphi = np.cos(sphere_mean[..., 1])
    Jsintheta = np.sin(sphere_mean[..., 0])
    Jsinphi = np.sin(sphere_mean[..., 1])
    J = np.zeros((sphere_mean.shape[0], sphere_mean.shape[1], 3, 2))
    J[..., 0, 0] = Jcostheta * Jcosphi
    J[..., 0, 1] = -Jsintheta * Jsinphi
    J[..., 1, 0] = Jcostheta * Jsinphi
    J[..., 1, 1] = Jsintheta * Jcosphi
    J[..., 2, 0] = -Jsintheta
    return J


def two_view_covariance(
    norm1,
    norm2,
    noise,
    var1=None,
    var2=None,
    two_view_covar_approach=covar_sphere_thorough_spherical_mean,
    prior_std_multiplier=None,
    lc_std_multiplier=None,
):
    """Compute the covariance matrix of two normals."""
    Covar_sphere, norm_sphere_mean, _ = two_view_covar_approach(norm1, norm2)
    J = Jacobian(norm_sphere_mean)

    _, R = np.linalg.eigh(Covar_sphere)
    A = R.transpose((0, 1, 3, 2)) @ Covar_sphere @ R
    A[..., 0, 0] = np.maximum(A[..., 0, 0], noise)
    A[..., 1, 1] = np.maximum(A[..., 1, 1], noise)
    Covar_sphere = R @ A @ R.transpose((0, 1, 3, 2))
    if lc_std_multiplier is not None:
        Covar_sphere *= lc_std_multiplier**2
    if prior_std_multiplier is not None:
        if var1 is not None:
            var1 *= prior_std_multiplier**2
        if var2 is not None:
            var2 *= prior_std_multiplier**2
    if var1 is not None:
        Covar_sphere[..., 0, 0] = np.maximum(Covar_sphere[..., 0, 0], var1)
        Covar_sphere[..., 1, 1] = np.maximum(Covar_sphere[..., 1, 1], var1)
    if var2 is not None:
        Covar_sphere[..., 0, 0] = np.maximum(Covar_sphere[..., 0, 0], var2)
        Covar_sphere[..., 1, 1] = np.maximum(Covar_sphere[..., 1, 1], var2)

    Cov_cartesian = J @ Covar_sphere @ J.transpose((0, 1, 3, 2))
    Cov_cartesian[..., 0, 0] = Cov_cartesian[..., 0, 0].clip(0)
    Cov_cartesian[..., 1, 1] = Cov_cartesian[..., 1, 1].clip(0)
    Cov_cartesian[..., 2, 2] = Cov_cartesian[..., 2, 2].clip(0)
    return Cov_cartesian


class Normals(BaseClass, PriorUtils):
    """Class for image normals."""

    default_conf = {
        "inherent_polar_noise": np.pi / 180,
        "std_multiplier": 1,
        "lc_std_multiplier": 1,
        "prior_std_multiplier": 1,
        "downscale_factor": 2,
        "prior_uncertainty": True,  # uncertainties predicted by the model
        "flip_consistency": False,  # uncertainties via computing difference between the normals
        "verbose": 0,
        # estimates of the imaga and fipped image
    }

    def _init(self, normals_dict, camera, mask=None, continuity_mask=None, **kwargs):
        if self.conf.flip_consistency:
            assert "normals2" in normals_dict, "Second normals must be provided for loop consistency"
        if self.conf.std_multiplier != 1:
            assert (
                self.conf.lc_std_multiplier == 1 and self.conf.prior_std_multiplier == 1
            ), "If std_multiplier is not 1, then lc_std_multiplier and prior_std_multiplier must be 1"

        normals_dict["normals"] = cv2.resize(normals_dict["normals"], (camera.int_width, camera.int_height))
        normals_dict["normals"] /= np.linalg.norm(normals_dict["normals"], axis=-1, keepdims=True)
        H, W = normals_dict["normals"].shape[:-1]
        d1 = cv2.resize(
            normals_dict["normals"], (int(W // self.conf.downscale_factor), int(H // self.conf.downscale_factor))
        )
        d1 /= np.linalg.norm(d1, axis=-1, keepdims=True)
        if self.conf.flip_consistency:
            normals_dict["normals2"] = cv2.resize(normals_dict["normals2"], (camera.int_width, camera.int_height))
            normals_dict["normals2"] /= np.linalg.norm(normals_dict["normals2"], axis=-1, keepdims=True)
            d2 = cv2.resize(
                normals_dict["normals2"],
                (int(W // self.conf.downscale_factor), int(H // self.conf.downscale_factor)),
            )
            d2 /= np.linalg.norm(d2, axis=-1, keepdims=True)
        if "normals_variance" in normals_dict:
            normals_dict["normals_variance"] = cv2.resize(
                normals_dict["normals_variance"], (camera.int_width, camera.int_height)
            )
            dv1 = cv2.resize(
                normals_dict["normals_variance"],
                (int(W // self.conf.downscale_factor), int(H // self.conf.downscale_factor)),
            )
            if self.conf.flip_consistency:
                normals_dict["normals2_variance"] = cv2.resize(
                    normals_dict["normals2_variance"], (camera.int_width, camera.int_height)
                )
                dv2 = cv2.resize(
                    normals_dict["normals2_variance"],
                    (int(W // self.conf.downscale_factor), int(H // self.conf.downscale_factor)),
                )

        self.data = normals_dict["normals"]
        if self.conf.flip_consistency:
            self.data = (normals_dict["normals2"] + self.data) / 2
            self.data /= np.linalg.norm(self.data, axis=-1, keepdims=True)
            self.data_downscaled = (d1 + d2) / 2
            self.data_downscaled /= np.linalg.norm(self.data_downscaled, axis=-1, keepdims=True)
            if self.conf.prior_uncertainty:
                var1 = normals_dict["normals_variance"]
                vard1 = dv1
                var2 = normals_dict.get("normals2_variance", None)
                vard2 = dv2
            else:
                var1 = None
                vard1 = None
                var2 = None
                vard2 = None
            self.uncertainty = two_view_covariance(
                normals_dict["normals"],
                normals_dict["normals2"],
                var1=var1,
                var2=var2,
                noise=self.conf.inherent_polar_noise,
                prior_std_multiplier=self.conf.prior_std_multiplier,
                lc_std_multiplier=self.conf.lc_std_multiplier,
            )
            self.uncertainty_downscaled = two_view_covariance(
                d1,
                d2,
                var1=vard1,
                var2=vard2,
                noise=self.conf.inherent_polar_noise,
                prior_std_multiplier=self.conf.prior_std_multiplier,
                lc_std_multiplier=self.conf.lc_std_multiplier,
            )
        else:
            v = normals_dict["normals_variance"]
            Covar_sphere = np.zeros((H, W, 2, 2), dtype=np.float32)
            Covar_sphere[..., 0, 0] = Covar_sphere[..., 1, 1] = v
            J = Jacobian(self.data)
            self.uncertainty = J @ Covar_sphere @ J.transpose((0, 1, 3, 2))
            Covar_sphere_downscaled = np.zeros((*dv1.shape, 2, 2), dtype=np.float32)
            Covar_sphere_downscaled[..., 0, 0] = Covar_sphere_downscaled[..., 1, 1] = dv1
            J = Jacobian(d1)
            self.uncertainty_downscaled = J @ Covar_sphere_downscaled @ J.transpose((0, 1, 3, 2))
            self.data_downscaled = d1

        self.uncertainty = self.uncertainty * (self.conf.std_multiplier**2)
        self.uncertainty_downscaled = self.uncertainty_downscaled * (self.conf.std_multiplier**2)

        if mask is not None and self.uncertainty is not None:
            if mask.shape != self.uncertainty.shape:
                mask = (
                    cv2.resize(
                        mask.astype(np.float32), self.uncertainty.shape[:2][::-1], interpolation=cv2.INTER_NEAREST
                    )
                    > 0.5
                )

            self.uncertainty[~mask] = large_number
        if continuity_mask is not None:
            self.uncertainty[~continuity_mask] = large_number
