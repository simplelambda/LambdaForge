"""Per-device expiring circuits use actual observation failures, never infer ECC/Xid."""

from types import SimpleNamespace

from lambdaforge.runtime.GPUDeviceHealth import reconcile_devices, suppressed_devices
from lambdaforge.runtime.GPUObservation import collect_nvml


def row(uuid="GPU-a", **values):
    return {
        "uuid": uuid,
        "index": 0,
        "quality": "measured",
        "free_bytes": 40,
        "total_bytes": 80,
        **values,
    }


def test_failed_device_backoff_is_scoped_and_a_healthy_observation_closes_it():
    previous = [row(), row("GPU-b")]
    current = [row(quality="unavailable", error="NVMLError_GpuIsLost"), row("GPU-b")]
    output = reconcile_devices(previous, current, now=100, initial=5, maximum=30)
    assert output[0]["circuit"]["retry_after"] == 105
    assert output[0]["circuit"]["reason"] == "NVMLError_GpuIsLost"
    assert output[1]["circuit"]["state"] == "closed"
    assert len(suppressed_devices(output, 101)) == 1
    assert suppressed_devices(output, 105) == []
    recovered = reconcile_devices(output, [row(), row("GPU-b")], now=105, initial=5, maximum=30)
    assert recovered[0]["circuit"]["state"] == "closed"


def test_suppressed_reads_do_not_inflate_failure_counts():
    failed = reconcile_devices(
        [row()], [row(quality="unavailable")], now=100, initial=5, maximum=30
    )
    output = reconcile_devices(
        failed, [row(quality="unavailable", suppressed=True)], now=101, initial=5, maximum=30
    )
    assert output[0]["circuit"]["failures"] == 1


def test_missing_uuid_is_not_replaced_by_new_physical_device():
    output = reconcile_devices([row()], [row("GPU-new")], now=100, initial=5, maximum=30)
    by_uuid = {value["uuid"]: value for value in output}
    assert by_uuid["GPU-a"]["quality"] == "unavailable"
    assert by_uuid["GPU-a"]["free_bytes"] is None
    assert by_uuid["GPU-a"]["circuit"]["reason"] == "device_disappeared"
    assert "Xid" not in str(output) and "ECC" not in str(output)


def test_blocked_nvml_memory_is_not_requeried_until_device_backoff():
    calls = []

    class NVML:
        nvmlInit = staticmethod(lambda: None)
        nvmlShutdown = staticmethod(lambda: None)
        nvmlDeviceGetCount = staticmethod(lambda: 2)
        nvmlDeviceGetHandleByIndex = staticmethod(lambda i: i)
        nvmlDeviceGetUUID = staticmethod(lambda i: f"GPU-{i}")
        nvmlDeviceGetName = staticmethod(lambda i: "fake")

        def nvmlDeviceGetMemoryInfo(self, i):
            calls.append(i)
            return SimpleNamespace(free=40, total=80)

    output = collect_nvml(NVML(), suppressed=[{"uuid": "GPU-0", "index": 0}])
    assert calls == [1]
    assert output[0]["quality"] == "unavailable" and output[0]["suppressed"]
    assert output[1]["quality"] == "measured"
