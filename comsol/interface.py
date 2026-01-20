"""
COMSOL Multiphysics interface using mph library.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

try:
    import mph
except ImportError:
    mph = None

from electrode_generator.config import COMSOLConfig


class COMSOLInterface:
    """
    Interface to COMSOL Multiphysics via mph library.

    Handles:
    - Starting/stopping COMSOL server
    - Model creation and manipulation
    - Mesh import
    - Material property assignment
    - Running simulations
    """

    def __init__(self, config: Optional[COMSOLConfig] = None):
        """
        Initialize COMSOL interface.

        Args:
            config: COMSOL configuration
        """
        self.config = config or COMSOLConfig()
        self.client = None
        self.model = None

        if mph is None:
            raise ImportError(
                "mph library is required for COMSOL integration. "
                "Install with: pip install mph"
            )

    def connect(self, cores: Optional[int] = None) -> None:
        """
        Connect to COMSOL server.

        Args:
            cores: Number of CPU cores to use
        """
        cores = cores or self.config.num_cores

        try:
            self.client = mph.start(cores=cores)
            print(f"Connected to COMSOL server with {cores} cores")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to COMSOL: {e}")

    def disconnect(self) -> None:
        """Disconnect from COMSOL server."""
        if self.client is not None:
            self.client.disconnect()
            self.client = None
            print("Disconnected from COMSOL server")

    def create_model(self, name: str = "ElectrodeMicrostructure") -> Any:
        """
        Create a new COMSOL model.

        Args:
            name: Model name

        Returns:
            COMSOL model object
        """
        if self.client is None:
            self.connect()

        self.model = self.client.create(name)
        return self.model

    def load_model(self, path: Union[str, Path]) -> Any:
        """
        Load existing COMSOL model.

        Args:
            path: Path to .mph file

        Returns:
            COMSOL model object
        """
        if self.client is None:
            self.connect()

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        self.model = self.client.load(str(path))
        return self.model

    def save_model(self, path: Union[str, Path]) -> None:
        """
        Save current model.

        Args:
            path: Output path for .mph file
        """
        if self.model is None:
            raise RuntimeError("No model to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(str(path))
        print(f"Model saved to: {path}")

    def import_mesh(
        self,
        mesh_path: Union[str, Path],
        geometry_name: str = "imp1",
    ) -> None:
        """
        Import mesh file into COMSOL model.

        Args:
            mesh_path: Path to mesh file (STL, NASTRAN, etc.)
            geometry_name: Name for imported geometry
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Create or load a model first.")

        mesh_path = Path(mesh_path)
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

        # Get file format
        suffix = mesh_path.suffix.lower()

        if suffix == ".stl":
            self._import_stl(mesh_path, geometry_name)
        elif suffix in [".nas", ".nastran"]:
            self._import_nastran(mesh_path, geometry_name)
        else:
            raise ValueError(f"Unsupported mesh format: {suffix}")

    def _import_stl(self, path: Path, name: str) -> None:
        """Import STL file."""
        # Create import node
        imp = self.model / "geometry" / name
        imp.java.set("type", "stl")
        imp.java.set("filename", str(path))
        imp.java.importData()

    def _import_nastran(self, path: Path, name: str) -> None:
        """Import NASTRAN mesh file."""
        imp = self.model / "geometry" / name
        imp.java.set("type", "nastran")
        imp.java.set("filename", str(path))
        imp.java.importData()

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set model parameters.

        Args:
            parameters: Dictionary of parameter names and values
        """
        if self.model is None:
            raise RuntimeError("No model loaded")

        for name, value in parameters.items():
            self.model.parameter(name, str(value))

    def add_physics(self, physics_type: str, name: str) -> Any:
        """
        Add physics interface to model.

        Args:
            physics_type: COMSOL physics type identifier
            name: Name for the physics interface

        Returns:
            Physics interface object
        """
        if self.model is None:
            raise RuntimeError("No model loaded")

        physics = self.model.physics.create(physics_type, name)
        return physics

    def add_material(
        self,
        name: str,
        properties: Dict[str, float],
        domain: Optional[int] = None,
    ) -> Any:
        """
        Add material with properties.

        Args:
            name: Material name
            properties: Material property dictionary
            domain: Domain to assign material to

        Returns:
            Material object
        """
        if self.model is None:
            raise RuntimeError("No model loaded")

        material = self.model.material.create(name)

        for prop_name, value in properties.items():
            material.property(prop_name, str(value))

        if domain is not None:
            material.selection.set(domain)

        return material

    def mesh(self, size: str = "fine") -> None:
        """
        Generate mesh.

        Args:
            size: Mesh size preset ("extremely fine", "fine", "normal", "coarse")
        """
        if self.model is None:
            raise RuntimeError("No model loaded")

        mesh_node = self.model.mesh.create("mesh1")
        mesh_node.java.autoMeshSize(size)
        mesh_node.java.run()

    def solve(self, study_name: str = "std1") -> None:
        """
        Run simulation.

        Args:
            study_name: Name of study to run
        """
        if self.model is None:
            raise RuntimeError("No model loaded")

        study = self.model.study(study_name)
        study.run()

    def get_results(
        self,
        expression: str,
        dataset: str = "dset1",
    ) -> np.ndarray:
        """
        Extract results from simulation.

        Args:
            expression: COMSOL expression to evaluate
            dataset: Dataset name

        Returns:
            Result values as numpy array
        """
        if self.model is None:
            raise RuntimeError("No model loaded")

        result = self.model.evaluate(expression, dataset=dataset)
        return np.array(result)

    def export_results(
        self,
        output_path: Union[str, Path],
        format: str = "csv",
    ) -> None:
        """
        Export results to file.

        Args:
            output_path: Output file path
            format: Export format ("csv", "txt", "vtk")
        """
        if self.model is None:
            raise RuntimeError("No model loaded")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create export node
        exp = self.model / "export" / "data1"
        exp.java.set("filename", str(output_path))

        if format == "csv":
            exp.java.set("exporttype", "csv")
        elif format == "txt":
            exp.java.set("exporttype", "text")
        elif format == "vtk":
            exp.java.set("exporttype", "vtk")

        exp.java.run()

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False


class COMSOLModelBuilder:
    """
    High-level model builder for electrode simulations.
    """

    def __init__(self, interface: COMSOLInterface):
        """
        Initialize model builder.

        Args:
            interface: COMSOL interface instance
        """
        self.interface = interface
        self.model = None

    def create_electrode_model(
        self,
        mesh_path: Union[str, Path],
        electrode_type: str = "cathode",
    ) -> Any:
        """
        Create electrode simulation model.

        Args:
            mesh_path: Path to electrode mesh
            electrode_type: "cathode" or "anode"

        Returns:
            COMSOL model object
        """
        # Create base model
        self.model = self.interface.create_model(f"Electrode_{electrode_type}")

        # Set default parameters
        default_params = self._get_default_parameters(electrode_type)
        self.interface.set_parameters(default_params)

        # Import mesh
        self.interface.import_mesh(mesh_path)

        # Add materials
        self._add_electrode_materials(electrode_type)

        return self.model

    def _get_default_parameters(self, electrode_type: str) -> Dict[str, str]:
        """Get default parameters for electrode type."""
        params = {
            "T": "298.15[K]",  # Temperature
            "F": "96485[C/mol]",  # Faraday constant
            "R": "8.314[J/(mol*K)]",  # Gas constant
        }

        if electrode_type == "cathode":
            params.update({
                "c_max": "51765[mol/m^3]",  # Max Li concentration (NMC)
                "D_s": "1e-14[m^2/s]",  # Solid diffusion coefficient
                "k_0": "1e-11[m/s]",  # Reaction rate constant
                "sigma_s": "10[S/m]",  # Electronic conductivity
            })
        else:  # anode
            params.update({
                "c_max": "31507[mol/m^3]",  # Max Li concentration (Graphite)
                "D_s": "3.9e-14[m^2/s]",  # Solid diffusion coefficient
                "k_0": "5e-11[m/s]",  # Reaction rate constant
                "sigma_s": "100[S/m]",  # Electronic conductivity
            })

        return params

    def _add_electrode_materials(self, electrode_type: str) -> None:
        """Add materials for electrode simulation."""
        if electrode_type == "cathode":
            # Active material (NMC)
            self.interface.add_material(
                "NMC",
                {
                    "electricconductivity": "10[S/m]",
                    "D": "1e-14[m^2/s]",
                },
                domain=1,
            )
        else:
            # Active material (Graphite)
            self.interface.add_material(
                "Graphite",
                {
                    "electricconductivity": "100[S/m]",
                    "D": "3.9e-14[m^2/s]",
                },
                domain=1,
            )

        # Electrolyte
        self.interface.add_material(
            "Electrolyte",
            {
                "electricconductivity": "1[S/m]",
                "D": "1e-10[m^2/s]",
            },
            domain=0,
        )

    def setup_electrochemistry(self) -> None:
        """Set up electrochemistry physics."""
        if self.model is None:
            raise RuntimeError("No model created")

        # Add lithium-ion battery physics
        self.interface.add_physics("LithiumIonBattery", "liion")

    def setup_transport(self) -> None:
        """Set up transport physics."""
        if self.model is None:
            raise RuntimeError("No model created")

        # Add transport of diluted species
        self.interface.add_physics("TransportDilutedSpecies", "tds")

        # Add electric currents
        self.interface.add_physics("ElectricCurrents", "ec")

    def setup_study(
        self,
        study_type: str = "stationary",
        time_range: Optional[List[float]] = None,
    ) -> None:
        """
        Set up study for simulation.

        Args:
            study_type: "stationary" or "time_dependent"
            time_range: Time range for transient study [start, step, end]
        """
        if self.model is None:
            raise RuntimeError("No model created")

        study = self.model.study.create("std1")

        if study_type == "stationary":
            study.java.feature("st1").set("studyType", "stat")
        else:
            study.java.feature("st1").set("studyType", "time")
            if time_range:
                study.java.feature("st1").set("tlist", f"range({time_range[0]},{time_range[1]},{time_range[2]})")

    def run_and_export(
        self,
        output_dir: Union[str, Path],
        export_formats: List[str] = ["csv", "vtk"],
    ) -> Dict[str, Path]:
        """
        Run simulation and export results.

        Args:
            output_dir: Output directory
            export_formats: List of export formats

        Returns:
            Dictionary of exported file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run simulation
        self.interface.solve()

        # Export results
        exported = {}
        for fmt in export_formats:
            output_path = output_dir / f"results.{fmt}"
            self.interface.export_results(output_path, format=fmt)
            exported[fmt] = output_path

        return exported
