import os
import numpy as np
import SimpleITK as sitk
import logging
from typing import List, Dict, Any, Optional, Tuple

# Import from other modules in the package
from .registration import BrainRegistration, RegistrationParameters, ImageRegistration
from .image_utils import ImageUtils
from .surface_operations import SurfaceOperations
from .point_set_operations import PointSetOperations

# Set up logging
logger = logging.getLogger("SulciAnalysis")

IM_CENTER = 150  # surface coordinate + 150 = voxel index (pipeline convention)

# ---------------------------------------------------------------------------------------------
# Changes of November 2026 (see readme):
#  1. Result_averageSurf.vtk is now the TRUE average surface, built from the average image itself
#     (before: the reference specimen's surface moved with the FIRST specimen's deformation field,
#     i.e. the wrong specimen). The density maps (.nii) are unchanged; the .vtk density maps are
#     painted onto this true average surface.
#  2. Every specimen's deformation field is saved (fields/<specimen>_field.npz), so anything on a
#     specimen (vessels, landmarks, new curves) can be moved to the average later without
#     registering again. A run that stops can be restarted: saved fields are reused.
#  3. Every specimen's sulci and endocast surface moved onto the average are saved
#     (specimens/<specimen>_sulci_on_average.csv, specimens/<specimen>_surface_on_average.vtk).
#  4. The reference surface (ref_surface_file) is moved with ITS OWN specimen's field and saved
#     as Result_refSurf_on_average.vtk (optional, for comparison with the old surface).
#  5. Specimens can be registered in parallel (num_workers > 1).
#  6. All outputs go to output_dir (before: the folder the program was started from).
# ---------------------------------------------------------------------------------------------


def specimen_name(image_file: str) -> str:
    """Specimen name from its image file name (Pts_F_6034_OK.nii -> Pts_F_6034_OK)."""
    n = os.path.basename(image_file)
    for ext in ('.nii.gz', '.nii', '.mha', '.mhd', '.nrrd'):
        if n.lower().endswith(ext):
            return n[:-len(ext)]
    return os.path.splitext(n)[0]


def transport_array(points: np.ndarray, mx: np.ndarray, my: np.ndarray, mz: np.ndarray) -> np.ndarray:
    """Same rule as PointSetOperations.transport_points (point + displacement at the nearest voxel),
    for an (n, 3) array of voxel-index coordinates."""
    nx, ny, nz = mx.shape
    i = np.clip((points[:, 0] + 0.5).astype(int), 0, nx - 1)
    j = np.clip((points[:, 1] + 0.5).astype(int), 0, ny - 1)
    k = np.clip((points[:, 2] + 0.5).astype(int), 0, nz - 1)
    return points + np.stack([mx[i, j, k], my[i, j, k], mz[i, j, k]], 1)


def _process_specimen(task: Dict[str, Any]) -> Tuple[int, List[List[float]]]:
    """
    One specimen: register it to the average (or reuse its saved field), move its sulci and its
    endocast surface onto the average, save everything. Module-level so it can run in a worker process.
    """
    i, name = task['index'], task['name']
    out_dir = task['output_dir']
    field_file = os.path.join(out_dir, 'fields', f'{name}_field.npz')
    curr_image, nx, ny, nz = ImageUtils.create_np_3d(task['image_file'])

    if os.path.exists(field_file):
        z = np.load(field_file)
        mx, my, mz = z['mx'], z['my'], z['mz']
        logger.info(f"Image {i + 1} ({name}): saved field found, registration skipped")
    else:
        logger.info(f"Processing image and point set {i + 1} ({name})")
        average_im = np.load(task['average_file'])
        p = task['reg_params']
        registration = BrainRegistration(RegistrationParameters(**p))
        zeros = lambda: np.zeros((nx, ny, nz))
        mx, my, mz = registration.register_images(curr_image, average_im, zeros(), zeros(), zeros(),
                                                  zeros(), zeros(), zeros())
        disp = np.sqrt(mx ** 2 + my ** 2 + mz ** 2)
        logger.info(f"Image {i + 1} ({name}): Max displacement = {disp.max():.4f}, "
                    f"Mean displacement = {disp.mean():.4f}")
        del disp, average_im
        mx, my, mz = mx.astype(np.float32), my.astype(np.float32), mz.astype(np.float32)
        tmp = field_file[:-4] + '_tmp.npz'
        np.savez_compressed(tmp, mx=mx, my=my, mz=mz)
        os.replace(tmp, field_file)           # only a complete file gets the final name
    # (the saved field is float32; the fresh and the reused case both transport with it, so a
    #  restarted run gives exactly the same result as an uninterrupted one)

    # Sulci
    points = np.genfromtxt(task['curve_file'], delimiter=',')
    def_points = PointSetOperations.transport_points(points.tolist(), mx, my, mz)
    np.savetxt(os.path.join(out_dir, 'specimens', f'{name}_sulci_on_average.csv'),
               np.asarray(def_points), delimiter=',', fmt='%.6f')

    # Endocast surface
    out_surf = os.path.join(out_dir, 'specimens', f'{name}_surface_on_average.vtk')
    if task.get('surface_file'):
        SurfaceOperations.transport_vtk_surf(task['surface_file'], out_surf, mx, my, mz,
                                             im_center_x=IM_CENTER, im_center_y=IM_CENTER, im_center_z=IM_CENTER)
    else:
        verts, faces = SurfaceOperations.surface_from_image(curr_image, task['surface_threshold'])
        SurfaceOperations.write_vtk_surface(out_surf, transport_array(verts, mx, my, mz), faces,
                                            im_center_x=IM_CENTER, im_center_y=IM_CENTER, im_center_z=IM_CENTER)
    return i, def_points


class SulciAnalysis:
    """Analysis of sulci variability across brain samples."""

    def __init__(self, registration_params: RegistrationParameters):
        """
        Initialize with registration parameters.

        Args:
            registration_params: Parameters for registration
        """
        self.reg_params = registration_params
        self.registration = BrainRegistration(registration_params)
        logger.info(f"Initialized sulci analysis with registration parameters: {self.reg_params}")

    def compute_sulci_variability(self, average_im: np.ndarray,
                                  list_input_image_files: List[str],
                                  list_input_curve_files: List[str],
                                  ref_surface_file: Optional[str] = None,
                                  output_dir: str = '.',
                                  num_workers: int = 1,
                                  input_surface_files: Optional[List[str]] = None,
                                  surface_threshold: float = 8.0) -> List[List[List[float]]]:
        """
        Compute sulci variability by transporting sulci to the average shape.

        Args:
            average_im: Average brain image
            list_input_image_files: List of input image files
            list_input_curve_files: List of input curve files (same order as the images)
            ref_surface_file: Optional reference surface (a specimen's .vtk in the image frame); moved
                with its own specimen's field to Result_refSurf_on_average.vtk
            output_dir: Folder for all outputs
            num_workers: Number of specimens registered at the same time (1 = one after the other)
            input_surface_files: Optional list of each specimen's endocast surface (.vtk, image frame,
                same order as the images). Without it, each surface is built from the specimen's image.
            surface_threshold: Image value that marks the blurred surface shell (for building surfaces)

        Returns:
            List of deformed point sets
        """
        try:
            # Validate inputs
            if len(list_input_image_files) != len(list_input_curve_files):
                raise ValueError("Number of input image files must match number of curve files")
            if input_surface_files and len(input_surface_files) != len(list_input_image_files):
                raise ValueError("Number of input surface files must match number of image files")
            if ref_surface_file and not os.path.exists(ref_surface_file):
                raise FileNotFoundError(f"Reference surface file not found: {ref_surface_file}")

            os.makedirs(os.path.join(output_dir, 'fields'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'specimens'), exist_ok=True)
            names = [specimen_name(f) for f in list_input_image_files]

            # 1) Read the sulci
            list_point_sets = []
            for input_curve_file in list_input_curve_files:
                try:
                    list_point_sets.append(np.genfromtxt(input_curve_file, delimiter=',').tolist())
                except Exception as e:
                    logger.error(f"Error reading curve file {input_curve_file}: {str(e)}")
                    raise

            # 2) Register every specimen to the average, transport its sulci and surface
            average_file = os.path.join(output_dir, 'fields', '_average_for_workers.npy')
            np.save(average_file, np.asarray(average_im, dtype=np.float32))
            p = self.reg_params
            tasks = [dict(index=i, name=names[i], image_file=list_input_image_files[i],
                          curve_file=list_input_curve_files[i],
                          surface_file=input_surface_files[i] if input_surface_files else None,
                          average_file=average_file, output_dir=output_dir,
                          reg_params=dict(v1=p.v1, v2=p.v2, delta_updates=p.delta_updates, it_nb=p.it_nb),
                          surface_threshold=surface_threshold)
                     for i in range(len(list_input_image_files))]
            results = {}
            if num_workers > 1:
                from concurrent.futures import ProcessPoolExecutor
                logger.info(f"Registering {len(tasks)} specimens, {num_workers} at a time")
                with ProcessPoolExecutor(max_workers=num_workers) as ex:
                    for i, def_points in ex.map(_process_specimen, tasks):
                        results[i] = def_points
                        logger.info(f"Specimen {i + 1}/{len(tasks)} ({names[i]}) done")
            else:
                for t in tasks:
                    i, def_points = _process_specimen(t)
                    results[i] = def_points
                    logger.info(f"Specimen {i + 1}/{len(tasks)} ({names[i]}) done")
            list_def_point_sets = [results[i] for i in range(len(tasks))]
            os.remove(average_file)

            # 3) The average surface: built from the average image itself (true average shape)
            logger.info("Computing average surface from the average image")
            avg_surf_file = os.path.join(output_dir, 'Result_averageSurf.vtk')
            verts, faces = SurfaceOperations.surface_from_image(np.asarray(average_im), surface_threshold)
            SurfaceOperations.write_vtk_surface(avg_surf_file, verts, faces,
                                                im_center_x=IM_CENTER, im_center_y=IM_CENTER, im_center_z=IM_CENTER)

            # Reference surface, moved with ITS OWN specimen's field (for comparison with older runs)
            if ref_surface_file:
                ref_name = specimen_name(ref_surface_file)
                if ref_name in names:
                    z = np.load(os.path.join(output_dir, 'fields', f'{ref_name}_field.npz'))
                    SurfaceOperations.transport_vtk_surf(
                        ref_surface_file, os.path.join(output_dir, 'Result_refSurf_on_average.vtk'),
                        z['mx'], z['my'], z['mz'], im_center_x=IM_CENTER, im_center_y=IM_CENTER, im_center_z=IM_CENTER)
                else:
                    logger.warning(f"Reference surface {ref_name} does not match any specimen image name; "
                                   f"Result_refSurf_on_average.vtk not made")

            # 4) Save results
            logger.info("Saving initial points")
            im_point_sets = PointSetOperations.list_point_sets_to_image(
                list_point_sets, im_center_x=IM_CENTER, im_center_y=IM_CENTER, im_center_z=IM_CENTER)
            sitk.WriteImage(im_point_sets, os.path.join(output_dir, 'Result_InitPoints.nii'))

            logger.info("Saving deformed points")
            im_def_point_sets = PointSetOperations.list_point_sets_to_image(
                list_def_point_sets, im_center_x=IM_CENTER, im_center_y=IM_CENTER, im_center_z=IM_CENTER)
            sitk.WriteImage(im_def_point_sets, os.path.join(output_dir, 'Result_DefPoints.nii'))

            # One density map per label (.nii), painted onto the true average surface (.vtk)
            logger.info("Saving deformed points by label")
            PointSetOperations.list_point_sets_to_images_and_surfaces(
                list_def_point_sets, avg_surf_file, os.path.join(output_dir, 'Result'),
                im_center_x=IM_CENTER, im_center_y=IM_CENTER, im_center_z=IM_CENTER)

            logger.info("Sulci variability analysis complete")
            return list_def_point_sets

        except Exception as e:
            logger.error(f"Error in compute_sulci_variability: {str(e)}")
            raise
