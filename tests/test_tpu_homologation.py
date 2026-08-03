"""Testes dos critérios de homologação real do BL-08."""

from scripts.homologate_tpu import validate_tpu_result


def _valid_result():
    return {
        "experiment": {"status": "success"},
        "environment": {"device_type": "TPU"},
        "execution": {"xla_world_size": 8},
        "tpu_acceleration_check": {
            "accelerator_active": True,
            "xla_runtime_metrics": {
                "compile_count": 2,
                "execute_count": 20,
                "mark_step_count": 18,
            },
        },
    }


def test_accepts_complete_tpu_multicore_evidence():
    assert validate_tpu_result(_valid_result(), expected_cores=8) == []


def test_rejects_cpu_result():
    result = _valid_result()
    result["environment"]["device_type"] = "CPU"

    assert "device_type não é TPU" in " ".join(validate_tpu_result(result))


def test_rejects_wrong_world_size():
    result = _valid_result()
    result["execution"]["xla_world_size"] = 1

    assert "deve ser 8" in " ".join(validate_tpu_result(result))


def test_rejects_missing_xla_execution():
    result = _valid_result()
    result["tpu_acceleration_check"]["xla_runtime_metrics"]["execute_count"] = 0

    assert "nenhuma execução XLA" in " ".join(validate_tpu_result(result))