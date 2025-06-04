from pathlib import Path

import numpy as np
import pycolmap

from mpsfm.baseclass import BaseClass
from mpsfm.extraction import Extraction
from mpsfm.sfm.mapper.bundle_adjustment import Optimizer
from mpsfm.sfm.mapper.depthconsistency import DepthConsistencyChecker
from mpsfm.sfm.mapper.image_selection import ImageSelection
from mpsfm.sfm.mapper.registration import MpsfmRegistration
from mpsfm.sfm.mapper.triangulator import MpsfmTriangulator
from mpsfm.sfm.scene.correspondences import Correspondences
from mpsfm.sfm.scene.reconstruction import MpsfmReconstruction
from mpsfm.utils.tools import log_status


class MpsfmMapper(BaseClass):
    """Mapper class for MP-SfM."""

    colmap_options = {
        **pycolmap.IncrementalMapperOptions().todict(),
        **{
            key: val
            for key, val in pycolmap.IncrementalPipelineOptions().todict().items()
            if not isinstance(val, dict)
        },
    }
    del colmap_options["image_selection_method"]

    default_conf = {
        "verbose": 0,
        "dataset": {},
        "colmap_options": colmap_options
        | {
            "filter_min_tri_angle": 0.001,
            "init_min_tri_angle": 0.001,
            "local_ba_min_tri_angle": 0.001,  # explore
            "min_angle": 0.001,
            "ignore_two_view_tracks": False,
            "local_ba_num_images": 5,
        },
        # extraction
        "matches_mode": "sparse",  # sparse, dense, both combine with +
        "pairs_type": "retrieval",
        "masks": ["sky"],  # sky
        "extractors": {**Extraction.default_conf},
        "extract": [],
        # depth consistency
        "depth_consistency": True,
        "depth_consistency_init": False,  # explore
        "pre_fail": False,
        "dc_all_frames": False,
        "dc_num_frames": 5,
        "depth_consistency_checker": {**DepthConsistencyChecker.default_conf},
        # mapper classes
        "correspondences": {**Correspondences.default_conf},
        "registration": {**MpsfmRegistration.default_conf},
        "ba": {**Optimizer.default_conf},
        "triangulator": {**MpsfmTriangulator.default_conf},
        "reconstruction": {**MpsfmReconstruction.default_conf},
        "next_view": {**ImageSelection.default_conf},
        # mpsfm logic
        "integrate": True,
        "int_covs_every_iter": False,
        "int_covs": False,
        "final_robustification": 0.125,
        # dev
        "regular_resc": False,
        "filtall": False,
        "times_relax_init_thresh": 1,
    }

    prev_num_reg_images = 0
    prev_num_num_points3D = 0
    first_refinement = True

    def _assert_configs(self):
        match_mode_split = self.conf.matches_mode.split("+")
        for mode in match_mode_split:
            assert mode in [
                "sparse",
                "dense",
                "depth",
                "cache",
            ], f"Invalid matches mode {mode}. Must be one of ['sparse', 'dense', 'depth', 'cache']"
        assert any(
            m in ["sparse", "dense"] for m in match_mode_split
        ), f"At least one of 'sparse' or 'dense' must be in matches_mode, got {match_mode_split}"

        valid_extracts = {"d", "depth", "n", "normal", "s", "sky", "f", "features", "m", "matches", "r", "retrieval"}

        for extract in self.conf.extract:
            assert extract in valid_extracts, f"Invalid extract type {extract}. Must be one of {valid_extracts}"

    def _propagate_conf(self):
        self.conf.reconstruction.colmap_options = self.conf.colmap_options
        self.conf.registration.colmap_options = self.conf.colmap_options
        self.conf.next_view.colmap_options = self.conf.colmap_options
        self.conf.triangulator.colmap_options = self.conf.colmap_options

        self.conf.correspondences.matches_mode = self.conf.matches_mode
        self.conf.extractors.matches_mode = self.conf.matches_mode
        self.conf.extractors.dataset = self.conf.dataset
        self.conf.reconstruction.matches_mode = self.conf.matches_mode

        self.conf.correspondences.verbose = self.conf.verbose
        self.conf.depth_consistency_checker.verbose = self.conf.verbose
        self.conf.registration.verbose = self.conf.verbose
        self.conf.ba.verbose = self.conf.verbose
        self.conf.triangulator.verbose = self.conf.verbose
        self.conf.reconstruction.verbose = self.conf.verbose
        self.conf.next_view.verbose = self.conf.verbose
        self.conf.extractors.verbose = self.conf.verbose

    def _init(
        self,
        references,
        cache_dir,
        sfm_outputs_dir,
        models=None,
        extract_only=False,
        scene_parser=None,
        **kwargs,
    ):
        self.cache_dir = cache_dir
        self.scene_parser = scene_parser
        self.sfm_outputs_dir = sfm_outputs_dir
        if models is None:
            models = {}
        self.extractor = Extraction(
            self.conf.extractors,
            scene_parser=scene_parser,
            models=models,
            cache_dir=self.cache_dir,
            sfm_outputs_dir=self.sfm_outputs_dir,
            references=references,
            extract=self.conf.extract,
        )

        if self.conf.pairs_type == "retrieval":
            self.extractor.extract_retrieval()
        self.extractor.extract_pairs(self.conf.pairs_type)
        self.extractor.extract_pairwise()

        if "depth" in self.conf.matches_mode:
            self.extractor.extract_normals()
        else:
            self.extractor.extract_mono()

        if len(self.conf.masks) > 0:
            self.extractor.extract_masks(self.conf.masks)

        if extract_only:
            return

        self.mpsfm_rec = MpsfmReconstruction.initialize_from_reconstruction(
            self.conf.reconstruction, references=references, scene_parser=scene_parser
        )

        self.correspondences = Correspondences(
            self.conf.correspondences,
            mpsfm_rec=self.mpsfm_rec,
            extractor=self.extractor,
            sfm_outputs_dir=self.sfm_outputs_dir,
        )
        self.correspondences.populate()
        self.optimizer = Optimizer(
            self.conf.ba,
            self.mpsfm_rec,
            self.correspondences,
        )

        self.mpsfm_rec.correspondences = self.correspondences

        self.mpsfm_rec.obs = pycolmap.ObservationManager(self.mpsfm_rec.rec, self.correspondences.cg)

        self.nextview = ImageSelection(self.conf.next_view, self.mpsfm_rec, self.correspondences)

        self.triangulator = MpsfmTriangulator(self.conf.triangulator, self.mpsfm_rec, self.correspondences.cg)

        self.registration = MpsfmRegistration(
            self.conf.registration,
            self.mpsfm_rec,
            self.correspondences,
            self.triangulator,
            optimizer=self.optimizer,
        )

        self.depth_consistency_checker = DepthConsistencyChecker(
            self.conf.depth_consistency_checker,
            self.mpsfm_rec,
            self.correspondences,
        )

        self.mpsfm_rec.initialize_mono_maps(
            extraction_obj=self.extractor,
            sfm_output_dir=self.sfm_outputs_dir,
            pairs_pth=self.extractor.sfm_pairs_path,
        )
        self.mpsfm_rec.init_kps_info(extraction_obj=self.extractor)

    def deregister_image(self, imid):
        """Deregister image from reconstruction"""
        self.mpsfm_rec.obs.deregister_image(imid)

    def at_registration_failure(self):
        """Deregister image if registration fails"""
        if self.mpsfm_rec.images[self.nextview.candid].has_pose:
            self.deregister_image(self.nextview.candid)
        print("Failed to run post triangulation refinement")

    def at_init_failure(self, init_pair):
        for imid in init_pair:
            if self.mpsfm_rec.images[imid].has_pose:
                self.deregister_image(imid)
            self.mpsfm_rec.images[imid].depth.reset()

    def at_success(self):
        """Called when registration is successful. This is used to update the state of the mapper"""
        self.nextview.at_success()
        self.depth_consistency_checker.at_registration_success()
        self.registration.half_ap_min_inliers = 0
        for imid in self.mpsfm_rec.images:
            if self.mpsfm_rec.images[imid].has_pose:
                continue
            self.mpsfm_rec.images[imid].failed_normal_registration = False

    def at_failure(self, imid):
        """Called when registration fails. This is used to update the state of the mapper"""
        self.nextview.at_failure(imid)
        if not self.mpsfm_rec.images[imid].failed_dc_check:
            self.mpsfm_rec.images[imid].failed_normal_registration = True

    def __call__(self, exclude_init_pairs=None, **kwargs):
        if exclude_init_pairs is None:
            exclude_init_pairs = set()
        relax_thresh = 0

        while True:

            ranked_init_pairs = self.nextview.find_init_pairs(
                exclude_init_pairs=exclude_init_pairs,
            )

            if len(ranked_init_pairs) == 0:
                if relax_thresh > self.conf.times_relax_init_thresh:
                    self.log("Failed to find init pair")
                    return self.mpsfm_rec, False
                else:
                    self.log(f"No pairs found with {relax_thresh} relaxations")
                    relax_thresh += 1
                    continue
            relax_thresh += 1
            for init_pair in ranked_init_pairs:
                success = self.registration.register_and_triangulate_init_pair(*init_pair)
                if not success:
                    self.log(f"Failed to register and triangulate init pair {init_pair}")
                if success:
                    success = self.post_init_refinement()
                    if not success:
                        self.log(f"Failed post init refinement for {init_pair}")
                if success and self.conf.depth_consistency and self.conf.depth_consistency_init:
                    success = self.depth_consistency_checker.check_init_pair(init_pair)
                    if not success:
                        self.log(f"Failed depth consistency check for init pair {init_pair}")
                if success:
                    success = self.iterative_global_refinement()
                if not success:
                    exclude_init_pairs.add(init_pair)
                    self.at_init_failure(init_pair)
                    continue
                break
            if success:
                init_pair_ = list(init_pair)
                self.log(
                    f"Init pair found: {init_pair_}: {self.mpsfm_rec.images[init_pair_[0]].name} and "
                    f"{self.mpsfm_rec.images[init_pair_[1]].name}",
                    level=1,
                )
                break

        while True:

            if self.nextview.candid is not None:
                if not self.mpsfm_rec.images[self.nextview.candid].has_pose:
                    self.at_failure(self.nextview.candid)
                else:
                    self.at_success()

            if not self.conf.dc_all_frames and (
                len(self.nextview.freeze_imids) > 0
                and (
                    self.depth_consistency_checker.reg_batch_dc_times_failed >= self.conf.dc_num_frames
                    or self.depth_consistency_checker.reg_batch_dc_times_failed
                    == len(self.mpsfm_rec.images) - self.mpsfm_rec.num_reg_images()
                )
            ):
                if self.depth_consistency_checker.conf.depth_consistency_resample:
                    resample_imids = {
                        imid
                        for imid in self.nextview.freeze_imids
                        if self.mpsfm_rec.images[imid].dc_times_inliers_resampled == 1
                        and self.mpsfm_rec.images[imid].failed_dc_check
                    }
                    for imid in resample_imids:
                        self.mpsfm_rec.images[imid].failed_dc_check = False

                    success = self.nextview.next_image(list(resample_imids))
                    if not success:
                        self.log("Resample failed, setting all failed dc checks to True")
                        self.depth_consistency_checker.skip_dc_check = True
                        for imid in self.mpsfm_rec.images:
                            self.mpsfm_rec.images[imid].ignore_matches_AP = {}
                else:
                    success = False
            else:
                success = self.nextview.next_image()
            if not success:
                # if number of registered images is number of images, then we are done
                if self.mpsfm_rec.num_reg_images() == self.mpsfm_rec.num_images():
                    self.log("\nEnding mapper loop because all images are registered")
                    break
                # if we have reduced num inliers too many times, we end reconstruction early
                if self.registration.half_ap_min_inliers >= self.registration.conf.reduce_min_inliers_at_failure:
                    self.log("\nEnding mapper loop because reduced min inliers too many times")
                    break
                if not self.conf.depth_consistency or self.depth_consistency_checker.skip_dc_check:
                    self.registration.half_ap_min_inliers += 1
                    self.log("Halving AP inliers", self.registration.half_ap_min_inliers)
                    for image in self.mpsfm_rec.images.values():
                        image.failed_normal_registration = False
                elif (
                    self.depth_consistency_checker.depth_cons_thresh >= 1
                    or self.depth_consistency_checker.cons_thresh_times_increased >= 4
                ):
                    self.depth_consistency_checker.skip_dc_check = True
                else:
                    self.depth_consistency_checker.relax_thresholds()

                self.nextview.freeze_imids = {
                    imid for imid, image in self.mpsfm_rec.images.items() if image.failed_normal_registration
                }
                self.nextview.candid = None
                continue
            self.log(
                "dpeth cons thresh",
                self.depth_consistency_checker.depth_cons_thresh,
                "times increased",
                self.depth_consistency_checker.cons_thresh_times_increased,
                level=2,
            )
            if self.conf.verbose == 0:
                log_status(
                    len(self.mpsfm_rec.reg_image_ids()),
                    len(self.mpsfm_rec.images),
                    f"Registering and Optimizing Image {self.nextview.candid} "
                    f"{self.mpsfm_rec.images[self.nextview.candid].name}",
                )
            if self.conf.depth_consistency and self.conf.pre_fail:
                self.log("Depth consistency pre-fail check")
                if self.depth_consistency_checker.pre_fail(self.nextview.candid):
                    continue

            if self.mpsfm_rec.best_next_ref_imid is not None:
                ref_imids = self.find_local_bundle(self.mpsfm_rec.best_next_ref_imid, return_points=False)["optim_ids"]
            else:
                ref_imids = None
            if not self.registration.register_and_triangulate_next_image(self.nextview.candid, ref_imids=ref_imids):
                self.at_registration_failure()
                self.log(
                    f"Failed to register and triangulate next image {self.nextview.candid}:"
                    f"{self.mpsfm_rec.images[self.nextview.candid].name}"
                )
                continue
            if not self.post_registration_refinement(
                self.nextview.candid, check_depth_consistency=not self.depth_consistency_checker.skip_dc_check
            ):
                self.at_registration_failure()
                continue

            if not self.iterative_local_refinement(self.nextview.candid):
                self.at_registration_failure()
                continue

            if self.mpsfm_rec.num_reg_images() != self.mpsfm_rec.num_images() and self.check_run_global_refinement():
                if self.conf.verbose == 0:
                    log_status(len(self.mpsfm_rec.reg_image_ids()), len(self.mpsfm_rec.images), "GLOBAL REFINEMENT")
                if not self.iterative_global_refinement():
                    self.at_registration_failure()
                    continue

            if self.conf.verbose > 1 and self.mpsfm_rec.num_reg_images() % 5 == 0:
                self.visualization()
                input("Max verbose. Press enter to continue")

        if self.conf.verbose == 0:
            log_status(len(self.mpsfm_rec.reg_image_ids()), len(self.mpsfm_rec.images), "GLOBAL REFINEMENT")
        self.iterative_global_refinement(
            param_multiplier=self.conf.final_robustification if self.conf.final_robustification is not None else 1,
            final=True,
        )
        
        if self.conf.verbose > 1:
            print(50 * "-")
            print("Reconstruction Desc")
            for image in self.mpsfm_rec.registered_images.values():
                print(image.name)
                print(f"\t{image}")
                print(f"\t{image.cam_from_world}")
            print(50 * "-")
        return self.mpsfm_rec, True

    # --- Optimization Utils ---
    def _refinement(self, bundle, int_covs, mode="global", refimid=None, allow_scale_filter=False, **kwargs):
        """Runs iterative refinement"""
        self.on_BA_start(bundle, mode)
        _, success = self.adjust_bundle(
            bundle, int_covs, mode=mode, refimid=refimid, allow_scale_filter=allow_scale_filter, **kwargs
        )
        self.on_BA_end(bundle, mode)
        if not success:
            print("Failed to run ba")
            return None, False

        num_observations = len(bundle["pts3D"])
        num_changed_observations, filtered_imids = self.filter_bundle(bundle)

        num_changed_observations += self.triangulator.complete_and_merge_tracks(bundle["pts3D"])
        changed = 0 if num_observations == 0 else num_changed_observations / num_observations
        if len(filtered_imids) > 0:
            changed = "deregistered"
            return changed, False

        return changed, True

    def iterative_local_refinement(self, imid) -> bool:
        """Runs iterative local refinement"""
        if self.conf.verbose > 0:
            print(100 * "-")
            print(f"Starting local refinement for image {imid}: {self.mpsfm_rec.images[imid].name}")
            print(100 * "-")
        self.triangulator.complete_and_merge_all_tracks()
        self.first_refinement = True

        for it in range(self.conf.colmap_options.ba_local_max_refinements):
            self.log("Iteration:", it, level=1)
            local_bundle = self.find_local_bundle(imid)
            observed_bundle = self.find_subset_bundle(local_bundle)
            self.log("\tCalculating point covariances...", tstart=True, level=1)
            self.optimizer.calculate_point_covs(observed_bundle)
            self.log(tend=True, level=1)
            changed, success = self._refinement(
                local_bundle, int_covs=self.conf.int_covs, mode="local", refimid=imid, allow_scale_filter=True
            )

            if not success:
                if changed == "deregistered":
                    print("An image got deregistered during local refinement")
                    if not self.mpsfm_rec.images[imid].has_pose:
                        print(f"Image {imid} is got deregistered during refinement")
                        return False
                raise ValueError(f"Failed to run local refinement for image {imid}")
            if changed < self.conf.colmap_options.ba_local_max_refinement_change:
                break
            if not self.mpsfm_rec.images[imid].has_pose:
                print(f"WARNING: Image {imid} is got deregistered during refinement")
                return False
        return True

    def iterative_global_refinement(self, **kwargs):
        """Runs iterative global refinement"""
        if self.conf.verbose > 0:
            print(100 * "-")
            print("Starting global refinement")
            print(100 * "-")
        self.triangulator.complete_and_merge_all_tracks()

        self.first_refinement = True
        self.triangulator.retriangulate()
        if self.conf.filtall:
            self.filter_all()

        for it in range(self.conf.colmap_options.ba_global_max_refinements):
            self.log("Iteration:", it, level=1)

            bundle = self.find_global_bundle()
            self.optimizer.calculate_point_covs(bundle)
            if self.conf.regular_resc:
                shift_scale, success = self.optimizer.optimize_prior_shiftscale(bundle)
                self.log(f"Shift scale: {shift_scale}", level=1)
                self.mpsfm_rec.rescale_all(shift_scale)
            changed, success = self._refinement(
                bundle, int_covs=self.conf.int_covs, mode="global", allow_scale_filter=True, **kwargs
            )

            if not success:
                if changed is None:
                    print("Failed to run global refinement")
                    return False
                print("Image was deregistered during global refinement")
            self.mpsfm_rec.normalize()
            if changed == "deregistered" or changed < self.conf.colmap_options.ba_global_max_refinement_change:
                if "param_multiplier" in kwargs:  # just to make sure all experiments are consistent
                    continue
                break
        self.prev_num_reg_images = self.mpsfm_rec.num_reg_images()
        self.prev_num_num_points3D = self.mpsfm_rec.rec.num_points3D()
        return True

    def post_init_refinement(self):
        """Runs post init refinement"""
        self.first_refinement = True
        bundle = self.find_global_bundle()

        self.optimizer.calculate_point_covs(
            bundle
        )  # Here we want the P3d vars to include the depth info (even when clean covs???)
        shift_scale, success = self.optimizer.optimize_prior_shiftscale(bundle)
        if not success:
            print(f"Failed to optimize global shift scale for {bundle['optim_ids']}")
            return False
        self.log(f"Shift scale: {shift_scale}", level=2)
        self.mpsfm_rec.rescale_all(shift_scale)
        self.mpsfm_rec.activate_depths(bundle["optim_ids"])

        if not self.optimizer.refine_3d_points(bundle):
            print(f"Failed to refine global 3d points for {bundle['optim_ids']}")
            return False
        self.filter_all()
        if not self.mpsfm_rec.registered_images:
            print("Images got filtered during post init refinement")
            return False
        return True

    def post_registration_refinement(self, imid, check_depth_consistency=True):
        """Runs post registration refinement"""
        if self.conf.verbose > 0:
            print(100 * "-")
            print(
                f"Post registration refinement for {imid}: {self.mpsfm_rec.images[imid].name} \
                    ({len(self.mpsfm_rec.reg_image_ids())}/{len(self.mpsfm_rec.images)})"
            )
            print(100 * "-")
        self.first_refinement = True

        if self.mpsfm_rec.images[imid].depth.activated:
            self.mpsfm_rec.images[imid].depth.reset()
        local_bundle = self.find_local_bundle(imid)
        # before 3d poitns have been refined with depth, doesn't make sense to filter points with uncertainty
        if (
            not self.conf.depth_consistency or not check_depth_consistency
        ):  # if we have depth cosnistency checks, pre filtering here can leed to death loop
            _, filtered_imids = self.filter_bundle(local_bundle)
            if len(filtered_imids) > 0:
                print(f"ALERT: Images got filtered during post registration refinement of {imid}: {filtered_imids}")
                if imid in filtered_imids:
                    print("Image got filtered during post registration refinement")
                    return False

        self.log("Refining 3d points...", tstart=True, level=1)
        _, success = self.optimizer.refine_3d_points(
            local_bundle, depth_type="prior" if not self.conf.integrate else "update"
        )
        self.log(tend=True, level=1)
        if not success:
            print(f"Failed to refine 3d points for {local_bundle['optim_ids']}")
            return False
        local_bundle = self.find_local_bundle(imid)
        if not self.conf.depth_consistency or not check_depth_consistency:
            _, filtered_imids = self.filter_bundle(local_bundle)
            if len(filtered_imids) > 0:
                print(f"ALERT: Images got filtered during post registration refinement of {imid}: {filtered_imids}")
                return False
        observed_bundle = self.find_subset_bundle(local_bundle)
        self.log("Calculating point covariances...", tstart=True, level=1)
        self.optimizer.calculate_point_covs(observed_bundle)
        self.log(tend=True, level=1)
        self.log("Optimizing prior scale...", tstart=True, level=1)
        shift_scale, success = self.optimizer.optimize_prior_shiftscale(local_bundle, allow_metric_scale_filter=True)
        self.log(tend=True, level=1)

        if not success:
            print(f"Failed to optimize shift scale for {local_bundle['optim_ids']}")
            return False
        self.mpsfm_rec.rescale_all(shift_scale)
        self.mpsfm_rec.activate_depths(set([imid]))

        self.log("Refining 3d points...", level=1)
        if self.conf.integrate and (not self.integrate_bundle([imid], int_covs=self.conf.int_covs)):
            print("Failed to integrate bundle")
            return None, False
        if self.conf.depth_consistency and check_depth_consistency:
            dc_bundle = self.find_local_bundle(imid, num_images=5, return_points=False)

            if not self.depth_consistency_checker.check_image(imid, dc_bundle):
                return False
        self.log("Refining 3d points...", tstart=True, level=1)
        if not self.optimizer.refine_3d_points(
            local_bundle, depth_type="prior" if not self.conf.integrate else "update"
        ):
            print(f"Failed to refine global 3d points for {imid}")
            return False
        self.log(tend=True, level=1)
        local_bundle = self.find_local_bundle(imid)
        _, filtered_imids = self.filter_bundle(local_bundle)
        if len(filtered_imids) > 0:
            print(f"ALERT: Images got filtered during post registration refinement of {imid}: {filtered_imids}")
        if imid not in self.mpsfm_rec.registered_images:
            print("Images got filtered during post registration refinement")
            return False
        return True

    def integrate_bundle(self, imids, int_covs=True, cache_device="cpu", **kwargs):
        """Integrates bundle for given image ids"""
        for imid in imids:
            change = self.mpsfm_rec.images[imid].integrate(cache_device=cache_device)
            if int_covs and change and self.first_refinement:
                H = self.mpsfm_rec.images[imid].calculate_hessian(
                    downscaled=self.mpsfm_rec.images[imid].conf.downscaled,
                )
                self.mpsfm_rec.images[imid].calculate_int_covs_at_kps(H)

        if not self.conf.int_covs_every_iter:
            self.first_refinement = False
        return True

    def adjust_bundle(
        self, bundle: tuple, int_covs: bool, mode="global", refimid=None, allow_scale_filter=False, **kwargs
    ) -> tuple[dict, bool]:
        """Bundle adjustment for given bundle"""
        if self.conf.integrate:
            integrate_imids = bundle["optim_ids"] if mode == "global" else [refimid]
            cache_device = "cuda" if mode == "local" else "cpu"
            self.log("\tRefining depth maps...", tstart=True, level=1)
            if not self.integrate_bundle(integrate_imids, int_covs, cache_device=cache_device, **kwargs):
                print("Failed to integrate bundle")
                return None, False
            self.log(tend=True, level=1)
        if mode == "global":
            self.optimizer.update_truncation_multiplier(self.mpsfm_rec.reg_image_ids())
        self.log("\tAdjusting bunlde...", tstart=True, level=1)
        problem, success = self.optimizer.ba(bundle, mode=mode, allow_scale_filter=allow_scale_filter, **kwargs)
        self.log(tend=True, level=1)

        if not success:
            print("Failed to optimize bundle")
            return None, False
        return problem, True

    # --- Mapper Logic ---
    def check_run_global_refinement(self):
        """Checks if global refinement should be run."""
        num_reg_images = self.mpsfm_rec.rec.num_reg_images()
        num_points3D = self.mpsfm_rec.rec.num_points3D()
        thresh = 0.3
        return (
            (((num_reg_images - self.prev_num_reg_images) / self.prev_num_reg_images) > thresh)
            or ((num_reg_images - self.prev_num_reg_images) > 500)
            or (((num_points3D - self.prev_num_num_points3D) / self.prev_num_num_points3D) > thresh)
            or ((num_points3D - self.prev_num_num_points3D) > 250000)
        )

    def on_BA_start(self, bundle, mode):
        """Called before bundle adjustment starts. This is used to move images to GPU for local refinement"""
        if mode == "local" and self.conf.integrate:
            for imid in bundle["optim_ids"]:
                if not self.mpsfm_rec.images[imid].integrated:
                    continue
                self.mpsfm_rec.images[imid].move_to_device("cuda")

    def on_BA_end(self, bundle, mode):
        """Called after bundle adjustment ends. This is used to move images back to CPU for local refinement"""
        if mode == "local" and self.conf.integrate:
            for imid in bundle["optim_ids"]:
                if not self.mpsfm_rec.images[imid].integrated:
                    continue
                self.mpsfm_rec.images[imid].move_to_device("cpu")

    # --- Reconstruction Utils ---
    def filter_all(self):
        """Filters all points and images"""
        rec = self.mpsfm_rec
        rec.obs.filter_observations_with_negative_depth()
        filter_max_reproj_error = self.conf.colmap_options.filter_max_reproj_error
        filter_max_reproj_error *= np.median([image.kp_std for imid, image in rec.images.items()])

        num_changed_observations, _ = self.filter_all_points3D(
            filter_max_reproj_error, self.conf.colmap_options.filter_min_tri_angle
        )
        filtered_imids = self.filter_images()

        return num_changed_observations, filtered_imids

    def filter_bundle(self, bundle, filter_ims=True):
        """Filters 3D points and images in bundle"""
        rec = self.mpsfm_rec
        rec.obs.filter_observations_with_negative_depth()
        filter_max_reproj_error = self.conf.colmap_options.filter_max_reproj_error
        filter_max_reproj_error *= np.median([image.kp_std for imid, image in rec.images.items()])

        num_changed_observations = self.filter_local_points3D(
            bundle, filter_max_reproj_error, self.conf.colmap_options.filter_min_tri_angle
        )
        filtered_imids = self.filter_images() if filter_ims else None
        return num_changed_observations, filtered_imids

    def filter_images(self):
        obs = self.mpsfm_rec.obs
        registered_images_before = set(self.mpsfm_rec.registered_images.keys())
        obs.filter_images(
            self.conf.colmap_options.min_focal_length_ratio,
            self.conf.colmap_options.max_focal_length_ratio,
            self.conf.colmap_options.max_extra_param,
        )

        for imid, image in self.mpsfm_rec.registered_images.items():
            if image.num_points3D == 0:
                obs.deregister_image(imid)
        # manually returning filtered image ids
        filtered_images = registered_images_before - set(self.mpsfm_rec.registered_images.keys())
        return filtered_images

    def find_local_bundle(self, refimid, num_images=None, return_points=True):
        """Finds local bundle around image. image -1 is the reference image"""

        if num_images == 0 and not return_points:
            return {
                "optim_ids": set([refimid]),
            }
        if num_images is None:
            num_images = self.conf.colmap_options.local_ba_num_images
        rec = self.mpsfm_rec
        optim_imids = set(rec.find_local_bundle_ids(refimid, num_images)) | {refimid}
        out = {"ref_id": refimid, "optim_ids": optim_imids}

        if return_points:
            pts3D_list = [
                set(rec.images[imid].point3D_ids(rec.images[imid].get_observation_point2D_idxs()))
                for imid in optim_imids
            ]
            pts3D = set.union(*pts3D_list)
            out["pts3D"] = set(rec.images[refimid].point3D_ids(rec.images[refimid].get_observation_point2D_idxs()))

            out["constpoints"] = pts3D - out["pts3D"]
        return out

    def filter_all_points3D(self, filter_max_reproj_error, filter_min_tri_angle):
        """Filters all 3D points"""
        num_changed_observations = 0
        num_changed_observations += self.mpsfm_rec.obs.filter_all_points3D(
            filter_max_reproj_error,
            filter_min_tri_angle,
        )
        return num_changed_observations, True

    def find_invalid_depth_points(self, imids):
        """Finds invalid depth points for given image ids"""
        collect_risky_p3d = []
        for imid in imids:
            image = self.mpsfm_rec.images[imid]
            pt2D_ids = image.get_observation_point2D_idxs()
            kps_with3D = image.keypoint_coordinates(pt2D_ids)
            p3d_ids = np.array(image.point3D_ids(pt2D_ids))
            valid = image.depth.valid_at_kps(kps_with3D)
            collect_risky_p3d.append(set(p3d_ids[~valid]))
        return collect_risky_p3d

    def filter_local_points3D(
        self,
        local_bundle: tuple[set, set],
        filter_max_reproj_error: float,
        filter_min_tri_angle: float,
    ):
        """Filters 3D points in local bundle"""
        num_changed_observations = 0

        collect_risky_p3d = self.find_invalid_depth_points(local_bundle["optim_ids"])
        collect_risky_p3d = set.intersection(*collect_risky_p3d)
        pts3d = local_bundle["pts3D"]
        if "constpoints" in local_bundle:
            pts3d = pts3d.union(local_bundle["constpoints"])
        num_changed_observations += self.mpsfm_rec.obs.filter_points3D(
            filter_max_reproj_error,
            1.5,  # defualt colmap
            collect_risky_p3d,
        )
        num_changed_observations += self.mpsfm_rec.obs.filter_points3D(
            filter_max_reproj_error,
            filter_min_tri_angle,
            pts3d,
        )

        return num_changed_observations

    def find_global_bundle(self):
        """Get global bundle"""
        rec = self.mpsfm_rec
        optim_ids = set(imid for imid, image in rec.images.items() if image.has_pose)
        optim_3dpoints = set(rec.points3D.keys())
        return {"optim_ids": optim_ids, "pts3D": optim_3dpoints, "constpoints": set()}

    def find_subset_bundle(self, bundle):
        """Get subset bundle"""
        imids = bundle["optim_ids"]
        rec = self.mpsfm_rec
        optim_ids = set(imids)
        seen_3dpoints = set()
        for imid in imids:
            image = rec.images[imid]
            points2d = image.get_observation_point2D_idxs()
            points3d = image.point3D_ids(points2d)
            seen_3dpoints.update(points3d)
        for reg_imid, image in rec.registered_images.items():
            if reg_imid in imids:
                continue
            points2d = image.get_observation_point2D_idxs()
            points3D = set(image.point3D_ids(points2d))

            # check if any pf points3D are in seen_3dpoints
            if points3D.intersection(seen_3dpoints):
                optim_ids.add(reg_imid)
        return {"optim_ids": optim_ids, "pts3D": seen_3dpoints}

    # --- Visualization ---
    def visualization(self, *args, **kwargs):
        """Visualizes the reconstruction"""
        global_bundle = self.find_global_bundle()
        self.optimizer.calculate_point_covs(global_bundle)

        fig = self.mpsfm_rec.vis_depth_maps(self.scene_parser.rgb_dir)
        fig = self.mpsfm_rec.vis_cameras(fig)
        fig = self.mpsfm_rec.vis_colmap_points(fig)
        outfile = self.sfm_outputs_dir / "3d.html"
        self.log(f"Saving plotly visualization to {outfile}")
        fn = Path(outfile)
        fn.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(fn)


class MpsfmRefiner(MpsfmMapper):
    """Triangulator+refiner class for MP-SfM."""

    def __call__(self, **kwargs):
        self.mpsfm_rec.inheret_poses(self.scene_parser.rec)
        for imid in self.mpsfm_rec.images:
            self.registration.triangulate_image(imid)
        bundle = self.find_global_bundle()
        shift_scale = {}
        for imid in self.mpsfm_rec.images:
            bundle["ref_id"] = imid
            update, _ =  self.optimizer.optimize_prior_shiftscale(bundle)     
            shift_scale |= update
        self.mpsfm_rec.rescale_all(shift_scale)
        self.mpsfm_rec.activate_depths(bundle["optim_ids"])
        # trinauglation again -- depth maps influence triangulation
        for imid in self.mpsfm_rec.images:
            self.registration.triangulate_image(imid)
        self.filter_all()
        bundle = self.find_global_bundle()
        self.optimizer.calculate_point_covs(bundle)
        if not self.integrate_bundle(bundle["optim_ids"]):
                print("Failed to integrate bundle")
                return None, False
        return self.mpsfm_rec, True