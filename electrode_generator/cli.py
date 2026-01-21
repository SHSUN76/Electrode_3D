"""
Command-line interface for Electrode 3D Generator.
"""

import argparse
import logging
import sys
from pathlib import Path

from electrode_generator import ElectrodeGenerator, Config


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Electrode 3D Structure Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train SliceGAN model")
    train_parser.add_argument("image", type=str, help="Path to training image")
    train_parser.add_argument("--config", type=str, help="Path to config file")
    train_parser.add_argument("--epochs", type=int, help="Number of epochs")
    train_parser.add_argument("--output", type=str, help="Output directory")
    train_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate 3D structures")
    gen_parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    gen_parser.add_argument("--config", type=str, help="Path to config file")
    gen_parser.add_argument("--size", type=int, nargs=3, default=[64, 64, 64], help="Structure size")
    gen_parser.add_argument("--num", type=int, default=1, help="Number of structures")
    gen_parser.add_argument("--output", type=str, required=True, help="Output directory")
    gen_parser.add_argument("--seed", type=int, help="Random seed")
    gen_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # Pipeline command
    pipe_parser = subparsers.add_parser("pipeline", help="Run complete pipeline")
    pipe_parser.add_argument("image", type=str, help="Path to training image")
    pipe_parser.add_argument("--config", type=str, help="Path to config file")
    pipe_parser.add_argument("--epochs", type=int, help="Number of epochs")
    pipe_parser.add_argument("--num", type=int, default=1, help="Number of structures")
    pipe_parser.add_argument("--size", type=int, nargs=3, default=[64, 64, 64], help="Structure size")
    pipe_parser.add_argument("--output", type=str, required=True, help="Output directory")
    pipe_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # Metrics command
    metrics_parser = subparsers.add_parser("metrics", help="Calculate structure metrics")
    metrics_parser.add_argument("structure", type=str, help="Path to structure .npy file")
    metrics_parser.add_argument("--config", type=str, help="Path to config file")
    metrics_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export structure to mesh")
    export_parser.add_argument("structure", type=str, help="Path to structure .npy file")
    export_parser.add_argument("--output", type=str, required=True, help="Output file path")
    export_parser.add_argument("--format", type=str, choices=["stl", "obj"], default="stl")
    export_parser.add_argument("--config", type=str, help="Path to config file")
    export_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # Pipeline with refinement command
    refined_parser = subparsers.add_parser(
        "pipeline-refined",
        help="Run complete pipeline with Blender mesh refinement"
    )
    refined_parser.add_argument("input", type=str, help="Path to image or micro-CT folder")
    refined_parser.add_argument("--config", type=str, help="Path to config file")
    refined_parser.add_argument("--epochs", type=int, help="Number of epochs")
    refined_parser.add_argument("--num", type=int, default=1, help="Number of structures")
    refined_parser.add_argument("--size", type=int, nargs=3, default=[64, 64, 64], help="Structure size")
    refined_parser.add_argument("--output", type=str, required=True, help="Output directory")
    refined_parser.add_argument("--use-blender", action="store_true", default=True, help="Use Blender MCP")
    refined_parser.add_argument("--no-blender", action="store_true", help="Skip Blender refinement")
    refined_parser.add_argument("--voxel-size", type=float, default=2.0, help="Blender remesh voxel size")
    refined_parser.add_argument("--run-comsol", action="store_true", help="Run COMSOL simulation")
    refined_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # Load micro-CT command
    microct_parser = subparsers.add_parser("load-microct", help="Load micro-CT TIFF sequence")
    microct_parser.add_argument("folder", type=str, help="Path to folder containing TIFF slices")
    microct_parser.add_argument("--output", type=str, required=True, help="Output .npy file path")
    microct_parser.add_argument("--max-slices", type=int, help="Maximum number of slices to load")
    microct_parser.add_argument("--segment", action="store_true", help="Segment into phases")
    microct_parser.add_argument("--n-phases", type=int, default=3, help="Number of phases for segmentation")
    microct_parser.add_argument("--config", type=str, help="Path to config file")
    microct_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # Mesh refinement command
    refine_parser = subparsers.add_parser("refine-mesh", help="Generate Blender mesh refinement code")
    refine_parser.add_argument("mesh", type=str, help="Path to input mesh (STL/OBJ)")
    refine_parser.add_argument("--output", type=str, required=True, help="Output refined mesh path")
    refine_parser.add_argument("--voxel-size", type=float, default=2.0, help="Remesh voxel size")
    refine_parser.add_argument("--scale-comsol", action="store_true", help="Scale for COMSOL (mm to m)")
    refine_parser.add_argument("--code-output", type=str, help="Output Blender Python script path")
    refine_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # Export for COMSOL command
    comsol_export_parser = subparsers.add_parser("export-comsol", help="Export mesh for COMSOL")
    comsol_export_parser.add_argument("mesh", type=str, help="Path to input mesh (STL/OBJ)")
    comsol_export_parser.add_argument("--output", type=str, required=True, help="Output file path")
    comsol_export_parser.add_argument("--format", type=str, choices=["stl", "nastran", "gmsh"],
                                      default="nastran", help="Export format")
    comsol_export_parser.add_argument("--scale", type=float, default=0.001, help="Scale factor (default: mm to m)")
    comsol_export_parser.add_argument("--config", type=str, help="Path to config file")
    comsol_export_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    setup_logging(getattr(args, "verbose", False))
    logger = logging.getLogger(__name__)

    # Load config
    config = None
    if hasattr(args, "config") and args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    generator = ElectrodeGenerator(config)

    try:
        if args.command == "train":
            output_dir = args.output or "checkpoints"
            config.checkpoint_dir = Path(output_dir)
            history = generator.train(args.image, epochs=args.epochs)
            logger.info(f"Training completed. Final D_loss: {history['d_loss'][-1]:.4f}")

        elif args.command == "generate":
            generator.load_checkpoint(args.checkpoint)
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)

            import numpy as np

            structures = generator.generate(
                size=tuple(args.size),
                num_samples=args.num,
                seed=args.seed,
            )

            if args.num == 1:
                structures = [structures]

            for i, structure in enumerate(structures):
                # Save voxel
                np.save(output_dir / f"structure_{i:04d}.npy", structure)
                # Export meshes
                for fmt in config.postprocessing.export_formats:
                    generator.export_mesh(
                        structure,
                        output_dir / f"structure_{i:04d}.{fmt}",
                        format=fmt,
                    )

            logger.info(f"Generated {args.num} structure(s) to {output_dir}")

        elif args.command == "pipeline":
            results = generator.run_pipeline(
                image_path=args.image,
                output_dir=args.output,
                train=True,
                epochs=args.epochs,
                num_structures=args.num,
                structure_size=tuple(args.size),
            )
            logger.info(f"Pipeline completed. Metrics: {results['metrics']}")

        elif args.command == "metrics":
            import numpy as np

            structure = np.load(args.structure)
            metrics = generator.calculate_metrics(structure)
            print("\nMicrostructure Metrics:")
            print("-" * 40)
            for key, value in metrics.items():
                print(f"  {key}: {value:.6f}")

        elif args.command == "export":
            import numpy as np

            structure = np.load(args.structure)
            output_path = generator.export_mesh(
                structure,
                args.output,
                format=args.format,
            )
            logger.info(f"Exported mesh to: {output_path}")

        elif args.command == "pipeline-refined":
            use_blender = not args.no_blender

            results = generator.run_pipeline_with_refinement(
                input_path=args.input,
                output_dir=args.output,
                use_blender=use_blender,
                blender_voxel_size=args.voxel_size,
                run_comsol=args.run_comsol,
                train=True,
                epochs=args.epochs,
                num_structures=args.num,
                structure_size=tuple(args.size),
            )

            logger.info(f"Pipeline with refinement completed.")
            logger.info(f"Raw meshes: {results['raw_meshes']}")
            logger.info(f"Refined meshes: {results['refined_meshes']}")

            if use_blender and results['blender_codes']:
                logger.info("\n=== BLENDER REFINEMENT CODES GENERATED ===")
                logger.info("Execute via Blender MCP or manually in Blender.")
                logger.info(f"Code files saved to: {args.output}")

        elif args.command == "load-microct":
            import numpy as np
            from preprocessing import StackProcessor

            processor = StackProcessor(config.preprocessing)
            volume = processor.load_tiff_sequence(
                args.folder,
                max_slices=args.max_slices,
            )

            if args.segment:
                logger.info(f"Segmenting volume into {args.n_phases} phases...")
                volume = generator.segment_volume(volume, n_phases=args.n_phases)

            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, volume)

            logger.info(f"Saved volume: {volume.shape} to {output_path}")
            logger.info(f"Data type: {volume.dtype}")
            if args.segment:
                unique_phases = np.unique(volume)
                logger.info(f"Unique phases: {unique_phases}")

        elif args.command == "refine-mesh":
            from postprocessing import BlenderMeshRefiner

            refiner = BlenderMeshRefiner(voxel_size=args.voxel_size)

            blender_code = refiner.get_full_refinement_code(
                input_path=args.mesh,
                output_path=args.output,
                voxel_size=args.voxel_size,
                scale_for_comsol=args.scale_comsol,
            )

            # Save code to file
            code_path = Path(args.code_output) if args.code_output else Path(args.output).with_suffix('.py')
            with open(code_path, 'w') as f:
                f.write(blender_code)

            logger.info(f"Blender refinement code saved to: {code_path}")
            logger.info("\nTo execute, use Blender MCP:")
            logger.info("  mcp__blender__execute_blender_code(code=...)")
            logger.info("\nOr run in Blender Python console.")

            # Print the code for convenience
            print("\n" + "=" * 60)
            print("BLENDER PYTHON CODE:")
            print("=" * 60)
            print(blender_code)

        elif args.command == "export-comsol":
            import trimesh
            from postprocessing import MeshExporter

            mesh = trimesh.load(args.mesh)
            exporter = MeshExporter(config.postprocessing)

            # Apply scale
            mesh.apply_scale(args.scale)

            output_path = Path(args.output)

            if args.format == "nastran":
                exporter.export_nastran(mesh.vertices, mesh.faces, output_path)
            elif args.format == "gmsh":
                exporter.export_gmsh(mesh.vertices, mesh.faces, output_path)
            else:
                exporter.export(mesh, output_path, format="stl")

            logger.info(f"Exported for COMSOL: {output_path}")
            logger.info(f"Format: {args.format}")
            logger.info(f"Scale: {args.scale}")
            logger.info(f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
