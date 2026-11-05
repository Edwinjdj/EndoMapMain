import os
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as scim
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Import from other modules in the package
from .registration import BrainRegistration, RegistrationParameters, ImageRegistration
from .image_utils import ImageUtils

# Set up logging
logger = logging.getLogger("AverageShape")


@dataclass
class MeanEstimationParameters:
    """Parameters for mean shape estimation."""
    ref_im_nb: int  # Number of reference images
    it_nb: int = 20  # Number of iterations


class BrainShapeAnalysis:
    """Main class for brain shape analysis."""
    
    def __init__(self, mean_estim_params: MeanEstimationParameters, 
                registration_params: RegistrationParameters):
        """
        Initialize with parameters.
        
        Args:
            mean_estim_params: Parameters for mean shape estimation
            registration_params: Parameters for registration
        """
        self.mean_params = mean_estim_params
        self.reg_params = registration_params
        self.registration = BrainRegistration(registration_params)
        logger.info(f"Initialized brain shape analysis with parameters: "
                   f"mean_estim={self.mean_params}, reg={self.reg_params}")
        
    def _register_single_image(self, ref_image, current_template, iteration):
        """Register a single reference image to the current template."""
        nx, ny, nz = ref_image.shape
        
        # Create local displacement and force fields
        loc_reg_mx = np.zeros((nx, ny, nz))
        loc_reg_my = np.zeros((nx, ny, nz))
        loc_reg_mz = np.zeros((nx, ny, nz))
        
        loc_reg_fx = np.zeros((nx, ny, nz))
        loc_reg_fy = np.zeros((nx, ny, nz))
        loc_reg_fz = np.zeros((nx, ny, nz))
        
        # Register image
        curr_fx, curr_fy, curr_fz = self.registration.register_images(
            ref_image, current_template, loc_reg_mx, loc_reg_my, loc_reg_mz,
            loc_reg_fx, loc_reg_fy, loc_reg_fz
        )
        
        logger.info(f"Completed registration for image {iteration}")
        return curr_fx, curr_fy, curr_fz
    
    def compute_average_shape(self, input_image_files: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the average shape from a list of input images.
        
        Args:
            input_image_files: List of image file paths
            
        Returns:
            Average image and variability fields in x, y, z directions
        """
        try:
            # Validate input
            if not input_image_files:
                raise ValueError("No input image files provided")
            
            if len(input_image_files) < 2:
                raise ValueError("At least two input images are required")
            
            # Check if files exist
            for file_path in input_image_files:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Image file not found: {file_path}")
            
            logger.info(f"Computing average shape from {len(input_image_files)} images")
            
            # 1) Read data and allocate memory
            list_images = []
            for input_image_file in input_image_files:
                i_curr, nx, ny, nz = ImageUtils.create_np_3d(input_image_file)
                list_images.append(i_curr)
            
            i_mov = np.copy(list_images[0])
            
            # Initialize displacement fields
            mx = np.zeros((nx, ny, nz))
            my = np.zeros((nx, ny, nz))
            mz = np.zeros((nx, ny, nz))
            
            # Initialize force fields
            average_f_x = np.zeros((nx, ny, nz))
            average_f_y = np.zeros((nx, ny, nz))
            average_f_z = np.zeros((nx, ny, nz))
            
            # Initialize temporary fields for registration
            loc_reg_mx = np.zeros((nx, ny, nz))
            loc_reg_my = np.zeros((nx, ny, nz))
            loc_reg_mz = np.zeros((nx, ny, nz))
            
            loc_reg_fx = np.zeros((nx, ny, nz))
            loc_reg_fy = np.zeros((nx, ny, nz))
            loc_reg_fz = np.zeros((nx, ny, nz))
            
            # Initialize deformed image
            i_def = ImageRegistration.transport_image(i_mov, mx, my, mz)
            
            # 2) Iterative algorithm
            for k in range(self.mean_params.it_nb):
                if k % 1 == 0:
                    logger.info(f"Iteration: {k+1} / {self.mean_params.it_nb}")
                
                # Compute average forces
                loc_reg_mx, loc_reg_my, loc_reg_mz = self.registration.register_images(
                    list_images[0], i_def, loc_reg_mx, loc_reg_my, loc_reg_mz, 
                    loc_reg_fx, loc_reg_fy, loc_reg_fz
                )
                
                average_f_x = np.copy(loc_reg_mx)
                average_f_y = np.copy(loc_reg_my)
                average_f_z = np.copy(loc_reg_mz)
                
                # Determine number of workers (cores to use)
                # Leave one core free. Nov 2026: count the CPUs this process may actually use (on Curta,
                # cpu_count() gives the whole node, not the CPUs given to the job)
                try:
                    n_cpus = len(os.sched_getaffinity(0))
                except AttributeError:
                    n_cpus = multiprocessing.cpu_count()
                num_workers = max(1, n_cpus - 1)
                logger.info(f"Using {num_workers} workers for parallel processing")

                # Process images in parallel
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = []
                    for i, ref_image in enumerate(list_images[1:], 1):
                        logger.info(f"Submitting reference image {i}/{len(list_images)-1}")
                        future = executor.submit(self._register_single_image, ref_image, i_def, i)
                        futures.append(future)
    
                    # Collect results as they complete
                    for i, future in enumerate(as_completed(futures)):
                        try:
                            curr_fx, curr_fy, curr_fz = future.result()
                            average_f_x = average_f_x + curr_fx
                            average_f_y = average_f_y + curr_fy
                            average_f_z = average_f_z + curr_fz
                            logger.info(f"Processed result {i+1}/{len(futures)}")
                        except Exception as exc:
                            logger.error(f"Registration generated an exception: {exc}")
                
                # Normalize average forces
                average_f_x = average_f_x / len(list_images)
                average_f_y = average_f_y / len(list_images)
                average_f_z = average_f_z / len(list_images)
                
                # Auto-tuning of the update multiplicatory factor
                if k == 0:
                    curr_max_def = np.max(np.sqrt(average_f_x**2 + average_f_y**2 + average_f_z**2))
                    if curr_max_def > 1e-5:
                        p = 0.5 / curr_max_def
                    else:
                        p = 1.0
                
                # Update displacement fields
                mx = mx + p * average_f_x
                my = my + p * average_f_y
                mz = mz + p * average_f_z
                
                # Smooth displacement fields
                mx = scim.gaussian_filter(mx, self.reg_params.v2, mode='constant')
                my = scim.gaussian_filter(my, self.reg_params.v2, mode='constant')
                mz = scim.gaussian_filter(mz, self.reg_params.v2, mode='constant')
                
                # Compute deformed image
                i_def = ImageRegistration.transport_image(i_mov, mx, my, mz)
                
                # Log maximum displacement
                max_disp = np.max(np.sqrt(mx**2 + my**2 + mz**2))
                logger.info(f"Mean estimation iteration {k+1}: max displacement = {max_disp:.4f}")
            
            # 3) Compute variability around the average
            logger.info("Computing variability estimation")
            variability_x = np.zeros_like(average_f_x)
            variability_y = np.zeros_like(average_f_y)
            variability_z = np.zeros_like(average_f_z)
            
            for i, ref_image in enumerate(list_images[1:], 1):
                logger.info(f"Computing variability for image {i}/{len(list_images)-1}")
                
                # Reset temporary fields
                loc_reg_mx.fill(0.0)
                loc_reg_my.fill(0.0)
                loc_reg_mz.fill(0.0)
                
                curr_fx, curr_fy, curr_fz = self.registration.register_images(
                    ref_image, i_def, loc_reg_mx, loc_reg_my, loc_reg_mz,
                    loc_reg_fx, loc_reg_fy, loc_reg_fz
                )
                
                variability_x = variability_x + np.power(curr_fx - average_f_x, 2)
                variability_y = variability_y + np.power(curr_fy - average_f_y, 2)
                variability_z = variability_z + np.power(curr_fz - average_f_z, 2)
            
            # Calculate standard deviation (unbiased estimation)
            n = len(list_images) - 1  # Number of reference images excluding the first one
            divisor = max(1, n - 1)   # Avoid division by zero
            
            variability_x = np.sqrt(variability_x / divisor)
            variability_y = np.sqrt(variability_y / divisor)
            variability_z = np.sqrt(variability_z / divisor)
            
            logger.info("Average shape computation complete")
            
            return i_def, variability_x, variability_y, variability_z
        
        except Exception as e:
            logger.error(f"Error in compute_average_shape: {str(e)}")
            raise