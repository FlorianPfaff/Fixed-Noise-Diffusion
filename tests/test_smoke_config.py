from fixed_noise_diffusion.config import load_config


def test_smoke_config_loads():
    config = load_config("smoke.yaml")
    assert config["data"]["dataset"] == "fake"
    assert config["noise"]["mode"] == "fixed_pool"
    assert config["training"]["max_train_steps"] == 1


def test_old_config_path_alias_loads_packaged_config():
    config = load_config("configs/smoke.yaml")
    assert config["run_name"] == "smoke"
