#!/usr/bin/env python3
# brain_shape_analyzer.py - Main entry point for brain shape analysis

import json
import logging
import argparse
import numpy as np
import SimpleITK as sitk
from typing import Dict, List, Any, Tuple
import sys
import os
# Add the current directory to the path so Python can find your package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now use a regular import (not a relative one)
from brain_shape_analysis import (
    ImageUtils, BrainRegistration, BrainShapeAnalysis, SulciAnalysis,
    RegistrationParameters, MeanEstimationParameters, 
    PointSetOperations, SurfaceOperations,
    MemoryMonitor, BatchProcessor, optimize_numpy_memory,
    BrainVisualization, SulciAnalysisTools
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("brain_shape_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BrainShapeAnalyzer")


DEFAULT_WORKERS = 1  # specimens registered at the same time in Step 2 when the config has no "num_workers"


def get_num_workers(config: Dict[str, Any]) -> int:
    """Number of specimens registered at the same time in Step 2: config "num_workers", else the
    number of CPUs given to a Slurm job (Curta), else DEFAULT_WORKERS."""
    if config.get("num_workers"):
        return int(config["num_workers"])
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    return DEFAULT_WORKERS


def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in configuration file: {config_file}")
    
    # Validate required fields
    required_fields = ['input_image_files', 'output_dir']
    missing_fields = [field for field in required_fields if field not in config]
    
    if missing_fields:
        raise ValueError(f"Missing required fields in configuration: {', '.join(missing_fields)}")
    
    return config


def create_default_config(output_file: str) -> None:
    """
    Create a default configuration file.
    
    Args:
        output_file: Path to output configuration file
    """
    default_config = {
        "input_image_files": [
            "data/image1.nii",
            "data/image2.nii",
            "data/image3.nii"
        ],
        "input_curve_files": [
            "data/curves1.csv",
            "data/curves2.csv",
            "data/curves3.csv"
        ],
        "ref_surface_file": "data/reference_surface.vtk",
        "num_workers": 1,
        "surface_threshold": 8.0,
        "mean_estim_params": {
            "it_nb": 20
        },
        "registration_params": {
            "v1": 10.0,
            "v2": 1.0,
            "delta_updates": 0.5,
            "it_nb": 20
        },
        "memory_optimization": {
            "enable_memory_monitoring": True,
            "batch_processing": True,
            "max_batch_size": 5,
            "clear_unused_images": True
        },
        "output_dir": "results",
        "debug": {
            "save_intermediate_results": False,
            "visualization": {
                "enable": True,
                "save_images": True,
                "display_slices": False
            }
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    logger.info(f"Created default configuration file: {output_file}")


def analyze_average_shape(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Analyze the average shape of input images.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Average image and variability fields in x, y, z directions
    """
    # Set up parameters
    registration_params = RegistrationParameters(
        v1=config["registration_params"]["v1"],
        v2=config["registration_params"]["v2"],
        delta_updates=config["registration_params"]["delta_updates"],
        it_nb=config["registration_params"]["it_nb"]
    )
    
    mean_estim_params = MeanEstimationParameters(
        ref_im_nb=len(config["input_image_files"]),
        it_nb=config["mean_estim_params"]["it_nb"]
    )
    
    # Set up memory optimization if enabled
    memory_opts = config.get("memory_optimization", {})
    memory_monitor = None
    
    if memory_opts.get("enable_memory_monitoring", False):
        memory_monitor = MemoryMonitor(
            enable_monitoring=True,
            memory_threshold=memory_opts.get("memory_threshold", 0.85),
            clear_unused_images=memory_opts.get("clear_unused_images", True)
        )
    
    # Initialize brain shape analysis
    brain_analysis = BrainShapeAnalysis(mean_estim_params, registration_params)
    
    # Compute average shape
    logger.info("Starting average shape computation")
    average_im, variability_x, variability_y, variability_z = brain_analysis.compute_average_shape(
        config["input_image_files"]
    )
    
    # Save results
    logger.info("Saving average shape results")
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    tmp_im = sitk.ReadImage(config["input_image_files"][0])
    
    ImageUtils.save_np_array_as_image(average_im, tmp_im, 
                                    os.path.join(output_dir, "Result_defImage.nii"))
    ImageUtils.save_np_array_as_image(variability_x, tmp_im, 
                                    os.path.join(output_dir, "Result_Vx.nii"))
    ImageUtils.save_np_array_as_image(variability_y, tmp_im, 
                                    os.path.join(output_dir, "Result_Vy.nii"))
    ImageUtils.save_np_array_as_image(variability_z, tmp_im, 
                                    os.path.join(output_dir, "Result_Vz.nii"))
    
    # Create visualizations if enabled
    if config.get("debug", {}).get("visualization", {}).get("enable", False):
        viz_dir = os.path.join(output_dir, "visualizations")
        viz = BrainVisualization(viz_dir)
        viz.visualize_average_shape(average_im, variability_x, variability_y, variability_z)
    
    return average_im, variability_x, variability_y, variability_z


def analyze_sulci_variability(config: Dict[str, Any], average_im: np.ndarray) -> List[List[List[float]]]:
    """
    Analyze sulci variability.
    
    Args:
        config: Configuration dictionary
        average_im: Average shape image
        
    Returns:
        List of deformed point sets
    """
    # Check required inputs (Nov 2026: the reference surface is now optional; the average surface is
    # built from the average image)
    if not config.get("input_curve_files"):
        logger.warning("Skipping sulci variability analysis (missing curve files)")
        return []
    
    # Set up parameters
    registration_params = RegistrationParameters(
        v1=config["registration_params"]["v1"],
        v2=config["registration_params"]["v2"],
        delta_updates=config["registration_params"]["delta_updates"],
        it_nb=config["registration_params"]["it_nb"]
    )
    
    # Initialize sulci analysis
    sulci_analysis = SulciAnalysis(registration_params)
    
    # Compute sulci variability
    logger.info("Starting sulci variability analysis")
    list_def_point_sets = sulci_analysis.compute_sulci_variability(
        average_im,
        config["input_image_files"],
        config["input_curve_files"],
        config.get("ref_surface_file"),
        output_dir=config["output_dir"],
        num_workers=get_num_workers(config),
        input_surface_files=config.get("input_surface_files"),
        surface_threshold=config.get("surface_threshold", 8.0)
    )
    
    # Additional analysis and visualization
    if list_def_point_sets and config.get("debug", {}).get("visualization", {}).get("enable", False):
        output_dir = config["output_dir"]
        viz_dir = os.path.join(output_dir, "visualizations")
        
        # Compute sulci statistics
        logger.info("Computing sulci statistics")
        stats = SulciAnalysisTools.compute_sulci_statistics(list_def_point_sets)
        SulciAnalysisTools.generate_report(stats, os.path.join(output_dir, "sulci_report.txt"))
        
        # Create probability maps
        logger.info("Creating sulci probability maps")
        prob_maps = SulciAnalysisTools.create_labeled_volume(
            list_def_point_sets,
            image_size=300,
            smooth_sigma=5.0
        )
        
        # Visualize probability maps
        viz = BrainVisualization(viz_dir)
        viz.visualize_sulci_probability_maps(average_im, prob_maps)
        
        # Create 3D rendering
        try:
            viz.create_3d_rendering(average_im, prob_maps)
        except Exception as e:
            logger.warning(f"Failed to create 3D rendering: {str(e)}")
    
    return list_def_point_sets


def run_pipeline(config: Dict[str, Any]) -> None:
    """
    Run the complete analysis pipeline.
    
    Args:
        config: Configuration dictionary
    """
    try:
        # Create output directory
        output_dir = config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        # Save a copy of the configuration
        config_file = os.path.join(output_dir, "config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Step 1: Average shape analysis
        logger.info("Step 1: Average shape analysis")
        average_im, variability_x, variability_y, variability_z = analyze_average_shape(config)
        
        # Step 2: Sulci variability analysis
        logger.info("Step 2: Sulci variability analysis")
        list_def_point_sets = analyze_sulci_variability(config, average_im)
        
        logger.info("Analysis pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in analysis pipeline: {str(e)}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Brain Shape Analysis Tool")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Create config command
    create_config_parser = subparsers.add_parser('create-config', help='Create a default configuration file')
    create_config_parser.add_argument('output', help='Output configuration file path')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run the analysis pipeline')
    run_parser.add_argument('config', help='Configuration file path')
    
    # Average shape command
    avg_parser = subparsers.add_parser('average-shape', help='Compute average shape only')
    avg_parser.add_argument('config', help='Configuration file path')
    
    # Sulci variability command
    sulci_parser = subparsers.add_parser('sulci-variability', help='Compute sulci variability only')
    sulci_parser.add_argument('config', help='Configuration file path')
    sulci_parser.add_argument('--average-image', help='Path to pre-computed average image')
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        if args.command == 'create-config':
            create_default_config(args.output)
        
        elif args.command == 'run':
            config = load_config(args.config)
            run_pipeline(config)
        
        elif args.command == 'average-shape':
            config = load_config(args.config)
            analyze_average_shape(config)
        
        elif args.command == 'sulci-variability':
            config = load_config(args.config)
            
            if args.average_image:
                # Load pre-computed average image
                average_im, _, _, _ = ImageUtils.create_np_3d(args.average_image)
            else:
                # Compute average image
                average_im, _, _, _ = analyze_average_shape(config)
            
            analyze_sulci_variability(config, average_im)
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()