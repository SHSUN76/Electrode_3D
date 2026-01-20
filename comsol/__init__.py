"""
COMSOL Multiphysics integration module.

Provides:
- Mesh import/export for COMSOL
- Automated model setup
- Electrochemical simulation configuration
- Results extraction
"""

from comsol.interface import COMSOLInterface
from comsol.simulation import ElectrochemicalSimulation

__all__ = ["COMSOLInterface", "ElectrochemicalSimulation"]
