import numpy as np
import pandas as pd
import pytest

from fuzzy_characterization.config import load_config
from fuzzy_characterization.data.sample import SampleOptions, generate_sample_datasets
from fuzzy_characterization.data.schema import Datasets


@pytest.fixture(scope="session")
def sample_tables():
    return generate_sample_datasets(SampleOptions(n_users=150, days=21, seed=3))


@pytest.fixture(scope="session")
def datasets(sample_tables) -> Datasets:
    t = sample_tables
    ev = t["api_events"].copy()
    ev["timestamp"] = pd.to_datetime(ev["timestamp"])
    return Datasets(
        users=t["users"].copy(), events=ev, posts=t["posts"].copy(), comments=t["comments"].copy(),
        reactions=t["reactions"].copy(), streams=t["streams"].copy(), stream_permissions=t["stream_permissions"].copy(),
        group_memberships=t["group_memberships"].copy(), consent=t["consent"].copy(), user_id_col="user_id",
        metadata={"period_start": str(ev["timestamp"].min()), "period_end": str(ev["timestamp"].max())},
    )


@pytest.fixture(scope="session")
def quick_config():
    cfg = load_config("configs/quick_test.yaml")
    cfg.clustering.n_restarts = 3
    cfg.clustering.k_max = 5
    cfg.evaluation.stability_runs = 3
    cfg.output.root = "outputs/_tests"
    return cfg


@pytest.fixture
def rng():
    return np.random.default_rng(0)
