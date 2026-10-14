"""Public production backend contract.

The full contract lives in :mod:`unisim.backend.base`; this module preserves
the original public import path while benchmark metadata keeps its coarse
capability labels.
"""

from .backend.base import (
    COMPLETION_CONTRACT_VERSION,
    COMPLETION_OPERATIONS,
    CameraCfg,
    CompletionToken,
    DebugOverlayGetter,
    DebugPrimitive,
    HostBridgeTransferPlan,
    ImmediateCompletionToken,
    PhysicsStateLayout,
    PhysicsStateParts,
    PreStepControlOutput,
    SimBackend,
    TensorDataPlane,
    TensorExecution,
    TensorIOSpec,
    TensorLifecycleCapabilities,
    TensorProcessTopology,
    TensorRuntimeDiagnostic,
    tensor_device_matches,
    validate_debug_overlays,
    validate_tensor_device,
)
from .backend.packed_reset import (
    PACKED_RESET_HOST_ONLY_FIELDS,
    PACKED_RESET_MODEL_FIELDS,
    PACKED_RESET_PERSISTENT_CHANNELS,
    TorchPackedResetComposer,
)
from .errors import BackendCapability, BackendError, UnsupportedCapabilityError

__all__ = [
    "BackendCapability",
    "BackendError",
    "CameraCfg",
    "DebugOverlayGetter",
    "DebugPrimitive",
    "COMPLETION_CONTRACT_VERSION",
    "COMPLETION_OPERATIONS",
    "CompletionToken",
    "ImmediateCompletionToken",
    "PhysicsStateLayout",
    "PhysicsStateParts",
    "PreStepControlOutput",
    "SimBackend",
    "HostBridgeTransferPlan",
    "TensorExecution",
    "TensorDataPlane",
    "TensorProcessTopology",
    "TensorLifecycleCapabilities",
    "TensorRuntimeDiagnostic",
    "TensorIOSpec",
    "PACKED_RESET_HOST_ONLY_FIELDS",
    "PACKED_RESET_MODEL_FIELDS",
    "PACKED_RESET_PERSISTENT_CHANNELS",
    "TorchPackedResetComposer",
    "tensor_device_matches",
    "UnsupportedCapabilityError",
    "validate_debug_overlays",
    "validate_tensor_device",
]
