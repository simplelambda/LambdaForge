"""Direct host CPU/RAM/disk/GPU observation."""

from __future__ import annotations

import json

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ClusterResourceProbe import ClusterResourceProbe
from lambdaforge.controlplane.ResourceSnapshot import ResourceSnapshot
from lambdaforge.controlplane.Transport import Transport


class DirectClusterResourceProbe(ClusterResourceProbe):
    """Use standard Python, optional psutil and nvidia-smi without a system agent."""

    def probe(self, profile: ClusterProfile, transport: Transport) -> ResourceSnapshot:
        workspace = repr(profile.workspace)
        code = (
            "import json,os,shutil\n"
            "value={'cpu_total':os.cpu_count(),'cpu_load':None,'ram_total_bytes':None,"
            "'ram_available_bytes':None}\n"
            "try:\n"
            " import psutil\n"
            " value.update(cpu_load=psutil.cpu_percent(interval=0.1),"
            "ram_total_bytes=psutil.virtual_memory().total,"
            "ram_available_bytes=psutil.virtual_memory().available)\n"
            "except Exception: pass\n"
            f"path={workspace}\n"
            "while not os.path.exists(path):\n"
            " parent=os.path.dirname(path) or '.'\n"
            " if parent == path: break\n"
            " path=parent\n"
            "disk=shutil.disk_usage(path)\n"
            "value.update(disk_total_bytes=disk.total,disk_free_bytes=disk.free)\n"
            "print(json.dumps(value))\n"
        )
        result = transport.run((*profile.command_prefix, profile.python, "-c", code), timeout=20.0)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "Direct resource probe failed.")
        observed = json.loads(result.stdout)
        gpu = transport.run(
            (
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ),
            timeout=20.0,
        )
        gpus = []
        if gpu.returncode == 0:
            for line in gpu.stdout.splitlines():
                values = [item.strip() for item in line.split(",")]
                if len(values) == 5:
                    gpus.append(
                        {
                            "index": int(values[0]),
                            "name": values[1],
                            "utilization_percent": float(values[2]),
                            "memory_used_bytes": int(float(values[3]) * 1024**2),
                            "memory_total_bytes": int(float(values[4]) * 1024**2),
                        }
                    )
        observed["gpus"] = gpus
        available = {
            "cpu_cores": observed.get("cpu_total"),
            "ram_bytes": observed.get("ram_available_bytes"),
            "disk_bytes": observed.get("disk_free_bytes"),
            "gpu_count": sum(1 for gpu_value in gpus if gpu_value["utilization_percent"] == 0),
            "gpu_capacity_confidence": "observed-not-enforced",
        }
        return ResourceSnapshot(
            profile.name,
            True,
            profile.scheduler,
            observed,
            available,
            {"mode": "scheduler-managed" if profile.scheduler == "slurm" else "process-leases"},
        )
