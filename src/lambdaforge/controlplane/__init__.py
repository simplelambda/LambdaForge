"""Provider-neutral local control plane for clusters and persistent jobs."""

from lambdaforge.LazyExports import LazyExports

_DOCTOR_NAMES = ("Doctor", "DoctorCheck", "DoctorReport")
_JOB_NAMES = ("JobGroup", "JobHandle", "JobRecord", "JobState")
_PYTHON_RUNTIME_NAMES = (
    "NoCompatiblePythonRuntimeError",
    "PythonRuntime",
    "PythonRuntimePolicy",
    "PythonRuntimeRequirements",
)
_NAMES = (
    "ClusterAuthentication",
    "ClusterBootstrapResult",
    "ClusterCatalog",
    "ClusterProfile",
    "ClusterService",
    "ClusterStoragePolicy",
    "CommandResult",
    "ControlPlane",
    "ControlPlaneFactory",
    "CredentialProvider",
    "CredentialService",
    "CudaCompatibilityResolver",
    "EnvironmentCredentialProvider",
    "EnvironmentIdentity",
    "EnvironmentProvider",
    "ExecutionBundle",
    "ExecutionBundleBuilder",
    "ExecutionProfile",
    "ExistingEnvironmentProvider",
    "InteractiveCredentialProvider",
    "JobGroupStore",
    "JobService",
    "JobStore",
    "LocalScheduler",
    "LocalTransport",
    "ManagedEnvironmentProvider",
    "MultiClusterSubmissionService",
    "OverviewService",
    "PasswordSshTransport",
    "PreparedEnvironment",
    "ProcessIdentity",
    "ProcessScheduler",
    "ProcessSupervisor",
    "ProjectWheelBuilder",
    "ResourceService",
    "ResourceSnapshot",
    "Scheduler",
    "SchedulerCapabilities",
    "SchedulerCommand",
    "SchedulerSubmission",
    "SecretRedactor",
    "SlurmClusterResourceProbe",
    "SlurmProfile",
    "SlurmResourceMapping",
    "SlurmScheduler",
    "SshConnectionPolicy",
    "SshTransport",
    "StorageGcPlan",
    "StorageReport",
    "StorageService",
    "SystemKeyringCredentialProvider",
    "TorchInstallationPlan",
    "TorchInstallationPolicy",
    "Transport",
)

LazyExports.install(
    __name__,
    {
        **{name: ("lambdaforge.controlplane.Doctor", name) for name in _DOCTOR_NAMES},
        **{name: ("lambdaforge.controlplane.jobs", name) for name in _JOB_NAMES},
        **{
            name: ("lambdaforge.controlplane.python_runtime", name)
            for name in _PYTHON_RUNTIME_NAMES
        },
        "MicromambaArtifactStore": (
            "lambdaforge.controlplane.MicromambaArtifactStore",
            "MicromambaArtifactStore",
        ),
        "PythonRuntimeResolver": (
            "lambdaforge.controlplane.PythonRuntimeResolver",
            "PythonRuntimeResolver",
        ),
        **{name: (f"lambdaforge.controlplane.{name}", name) for name in _NAMES},
    },
)

__all__ = [
    *_DOCTOR_NAMES,
    *_JOB_NAMES,
    "MicromambaArtifactStore",
    "PythonRuntimeResolver",
    *_PYTHON_RUNTIME_NAMES,
    *_NAMES,
]
