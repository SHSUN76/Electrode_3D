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

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
