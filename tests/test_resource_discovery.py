"""Testes para gridsearch.resource_discovery — parâmetros de recurso do MILP PSLA4ML."""

import pytest

from gridsearch.resource_discovery import (
    DEFAULT_CLOUD_DISK_GB,
    GPU_CATALOG,
    LocalHardwareSpec,
    build_resource_catalog,
    build_resource_spec,
    communication_cost_matrix,
    detect_local_hardware,
    estimate_energy_rate_kwh_per_hour,
)


class TestDetectLocalHardware:
    def test_returns_positive_cpu_ram_disk(self):
        spec = detect_local_hardware()
        assert isinstance(spec, LocalHardwareSpec)
        assert spec.cpu_logical_cores >= 1
        assert spec.ram_total_gb > 0
        assert spec.disk_total_gb > 0
        assert spec.disk_free_gb >= 0

    def test_gpu_fields_are_none_or_consistent(self):
        spec = detect_local_hardware()
        # Sem CUDA: todos None. Com CUDA: todos preenchidos.
        gpu_fields = (spec.gpu_name, spec.gpu_vram_gb, spec.gpu_multiprocessor_count)
        assert all(f is None for f in gpu_fields) or all(f is not None for f in gpu_fields)


class TestGpuCatalog:
    @pytest.mark.parametrize("name", ["NVIDIA RTX 3090", "NVIDIA T4", "NVIDIA V100", "NVIDIA A100"])
    def test_known_gpus_present(self, name):
        assert name in GPU_CATALOG
        assert GPU_CATALOG[name]["fp16_tflops"] > 0
        assert GPU_CATALOG[name]["sm_count"] > 0


class TestBuildResourceSpec:
    def test_local_gpu_name_with_geforce_prefix_still_matches_catalog(self):
        """Regressao: torch reporta 'NVIDIA GeForce RTX 2050', catalogo usa 'NVIDIA RTX 2050'."""
        hw = LocalHardwareSpec(
            cpu_logical_cores=8, ram_total_gb=16.0, disk_total_gb=200.0, disk_free_gb=100.0,
            gpu_name="NVIDIA GeForce RTX 2050", gpu_vram_gb=4.0, gpu_multiprocessor_count=16,
        )
        spec = build_resource_spec("local", {"gpu": "NVIDIA RTX 3090", "cost_per_hour_usd": 0.04}, local_hardware=hw)
        assert spec["n_i"]["GPU"] == 16
        assert spec["g_i"]["GPU"] > 0  # nao deve cair para 0.0 por falha de matching

    def test_cloud_resource_uses_static_catalog(self):
        spec = build_resource_spec("colab", {"gpu": "NVIDIA T4", "cost_per_hour_usd": 0.0, "vram_gb": 16})
        assert spec["resource_id"] == "colab"
        assert spec["c_i"] == 0.0
        assert spec["m_i"] == 16
        assert spec["n_i"]["GPU"] == GPU_CATALOG["NVIDIA T4"]["sm_count"]
        assert spec["g_i"]["GPU"] > 0
        assert spec["source"] == "static_catalog"

    def test_cloud_resource_unknown_gpu_falls_back_to_default(self):
        spec = build_resource_spec("mystery_cloud", {"gpu": "Some Unknown GPU", "cost_per_hour_usd": 1.0})
        assert spec["source"] == "default"
        assert spec["d_i"] == DEFAULT_CLOUD_DISK_GB
        assert spec["n_i"] == {}
        assert spec["g_i"] == {}

    def test_local_resource_uses_detected_hardware(self):
        hw = LocalHardwareSpec(
            cpu_logical_cores=8, ram_total_gb=32.0, disk_total_gb=500.0, disk_free_gb=200.0,
            gpu_name="NVIDIA RTX 3090", gpu_vram_gb=24.0, gpu_multiprocessor_count=82,
        )
        spec = build_resource_spec("local", {"gpu": "NVIDIA RTX 3090", "cost_per_hour_usd": 0.04}, local_hardware=hw)
        assert spec["source"] == "detected_local"
        assert spec["m_i"] == 24.0
        assert spec["d_i"] == 200.0
        assert spec["n_i"]["GPU"] == 82


class TestEstimateEnergyRate:
    def test_computes_average_rate_for_environment(self):
        results = [
            {"status": "success", "selected_environment": "local",
             "resources": {"energy_kwh": 0.001, "train_time_sec": 3600}},
            {"status": "success", "selected_environment": "local",
             "resources": {"energy_kwh": 0.002, "train_time_sec": 3600}},
            {"status": "success", "selected_environment": "colab",
             "resources": {"energy_kwh": 0.0005, "train_time_sec": 3600}},
            {"status": "failed", "selected_environment": "local",
             "resources": {}},
        ]
        assert estimate_energy_rate_kwh_per_hour(results, "local") == pytest.approx(0.0015)
        assert estimate_energy_rate_kwh_per_hour(results, "colab") == pytest.approx(0.0005)

    def test_returns_none_without_samples(self):
        assert estimate_energy_rate_kwh_per_hour([], "local") is None
        assert estimate_energy_rate_kwh_per_hour(
            [{"status": "success", "selected_environment": "gcp", "resources": {"energy_kwh": 0.1, "train_time_sec": 3600}}],
            "local",
        ) is None


class TestBuildResourceCatalog:
    def test_builds_entry_per_environment(self):
        environments_details = {
            "local": {"gpu": "NVIDIA RTX 3090", "cost_per_hour_usd": 0.04},
            "colab": {"gpu": "NVIDIA T4", "cost_per_hour_usd": 0.0, "vram_gb": 16},
        }
        fake_hw = LocalHardwareSpec(
            cpu_logical_cores=8, ram_total_gb=32.0, disk_total_gb=500.0, disk_free_gb=200.0,
            gpu_name="NVIDIA RTX 3090", gpu_vram_gb=24.0, gpu_multiprocessor_count=82,
        )
        catalog = build_resource_catalog(environments_details, local_hardware=fake_hw)

        assert set(catalog.keys()) == {"local", "colab"}
        assert catalog["local"]["source"] == "detected_local"
        assert catalog["colab"]["source"] == "static_catalog"
        assert catalog["local"]["e_i"] is None  # sem histórico fornecido

    def test_populates_e_i_from_results_history(self):
        environments_details = {"local": {"gpu": "NVIDIA RTX 3090", "cost_per_hour_usd": 0.04}}
        results = [
            {"status": "success", "selected_environment": "local",
             "resources": {"energy_kwh": 0.0018, "train_time_sec": 3600}},
        ]
        fake_hw = LocalHardwareSpec(
            cpu_logical_cores=8, ram_total_gb=32.0, disk_total_gb=500.0, disk_free_gb=200.0,
            gpu_name="NVIDIA RTX 3090", gpu_vram_gb=24.0, gpu_multiprocessor_count=82,
        )
        catalog = build_resource_catalog(environments_details, results=results, local_hardware=fake_hw)
        assert catalog["local"]["e_i"] == pytest.approx(0.0018)


class TestCommunicationCostMatrix:
    def test_all_pairs_default_to_zero(self):
        matrix = communication_cost_matrix(["local", "colab", "gcp"])
        assert matrix == {
            ("local", "colab"): 0.0,
            ("local", "gcp"): 0.0,
            ("colab", "gcp"): 0.0,
        }

    def test_custom_default_value(self):
        matrix = communication_cost_matrix(["a", "b"], default=1.5)
        assert matrix == {("a", "b"): 1.5}

    def test_single_resource_has_no_pairs(self):
        assert communication_cost_matrix(["only_one"]) == {}
