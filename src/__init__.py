"""
Gravitational Slingshot Navigation (GSN-VLN)
=============================================

Vision-Language Navigation using phantom scalar field-induced repulsive metrics.

Core modules:
- dual_scalar_poisson_solver: Neural Poisson solver for φ+ and φ- fields
- conformal_metric: Conformal metric construction and geodesic shooting
- slingshot_policy: Main policy integrating all components
- openvla_wrapper: OpenVLA-7B with metric token injection
- affordance_extractor: Qwen2-VL for target/distractor extraction

Physics background:
From Brans-Dicke scalar-tensor theory with phantom scalar field,
we derive a dual scalar field model where:
  ∇²φ₊ = +4πG ρ_target   (attractive field)
  ∇²φ₋ = -4πG ρ_distractor (repulsive field, phantom)

The conformal metric is: g_ij = e^{2(φ₊ - λφ₋)} δ_ij
Geodesics in this metric naturally exhibit gravitational slingshot behavior,
being repelled by distractors while attracted to targets.

Authors: Slingshot-VLN Team
Target: IROS 2026
Project reconstructed by Claude according to latest IROS 2026 slingshot design - Nov 2025
"""

__version__ = "1.0.0"
__author__ = "Slingshot-VLN Team"
__target_conference__ = "IROS 2026"

from .dual_scalar_poisson_solver import DualScalarPoissonSolver
from .conformal_metric import ConformalMetric, GeodesicIntegrator
from .slingshot_policy import SlingshotPolicy
from .openvla_wrapper import OpenVLAWrapper
from .affordance_extractor import AffordanceExtractor

__all__ = [
    "DualScalarPoissonSolver",
    "ConformalMetric",
    "GeodesicIntegrator",
    "SlingshotPolicy",
    "OpenVLAWrapper",
    "AffordanceExtractor",
]
