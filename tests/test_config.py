import pytest

from flight_delay.config import Settings, deep_merge, load_settings, settings_from_dict

from conftest import ROOT


def test_base_config_loads_with_paper_defaults():
    s = load_settings(ROOT / "configs" / "base.yaml")
    assert s.scope.years == (2024, 2025)
    assert s.scope.delay_threshold_min == 30
    assert s.modeling.tuning.n_iter == 10
    assert s.scope.sample_tag == "thr30_n20000_rs42"


def test_test_profile_overrides_only_what_it_lists():
    s = load_settings([ROOT / "configs" / "base.yaml", ROOT / "configs" / "test.yaml"])
    assert s.scope.years == (2025,) and s.scope.months == (1, 2)
    assert s.modeling.cv_folds == 3 and s.modeling.tuning.n_iter == 3
    assert s.modeling.tuning.max_sample == 120_000  # untouched by the profile
    assert s.evaluation.n_boot == 30


def test_unknown_key_is_an_error():
    with pytest.raises(KeyError, match="unknown keys"):
        settings_from_dict({"scope": {"yeras": [2024]}})


def test_invalid_values_are_rejected():
    with pytest.raises(ValueError):
        settings_from_dict({"modeling": {"split_strategy": "por_dia"}})
    with pytest.raises(ValueError):
        settings_from_dict({"scope": {"months": [13]}})


def test_deep_merge_keeps_siblings():
    assert deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}}) == {"a": {"x": 1, "y": 3}}


def test_settings_are_immutable():
    s = Settings()
    with pytest.raises(AttributeError):
        s.scope.years = (2020,)
