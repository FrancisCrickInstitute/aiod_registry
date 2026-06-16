import json
import os
import sys

import pytest
import yaml
from pydantic import ValidationError

from aiod_registry.schema import ModelManifest, ModelParam
from aiod_registry.utils import (
    filter_empty_manifests,
    filter_location,
    flatten_manifest,
    generate_default_config,
    is_accessible,
    load_manifests,
    save_all_default_configs,
    _params_to_yaml,
)

# Example manifest data (based on cellpose.json)
EXAMPLE_MANIFEST = {
    "name": "Cellpose",
    "short_name": "cellpose",
    "metadata": {
        "description": "Cellpose is a generalist model for cell and nucleus segmentation.",
        "url": "https://cellpose.readthedocs.io/en/v3.1.1.1/",
        "repo": "https://github.com/MouseLand/cellpose",
        "pubs": [
            {
                "info": "Cellpose v1",
                "url": "https://doi.org/10.1038/s41592-020-01018-x",
                "title": "Cellpose: a generalist algorithm for cellular segmentation",
                "doi": "10.1038/s41592-020-01018-x",
                "authors": [
                    {
                        "name": "Carsen Stringer",
                        "affiliation": "HHMI Janelia Research Campus, Ashburn, VA, USA",
                    }
                ],
            }
        ],
    },
    "versions": {
        "cyto3": {
            "tasks": {
                "cyto": {
                    "locations": [
                        {"location": "https://www.cellpose.org/models/cyto3"},
                        {"location": "file:///nonexistent/path"},
                    ]
                }
            }
        },
        "cyto2": {"tasks": {"cyto": {"locations": [{"location": "file:///nonexistent/path"}]}}},
    },
    "params": [
        {
            "name": "Diameter",
            "arg_name": "diameter",
            "value": 0,
            "tooltip": "Diameter of the cells in pixels.",
        }
    ],
}


@pytest.fixture
def temp_manifest_file(tmp_path):
    manifest_path = tmp_path / "test_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(EXAMPLE_MANIFEST, f)
    return manifest_path


def test_load_manifests_basic(temp_manifest_file):
    manifests = load_manifests(paths=[temp_manifest_file])
    assert "cellpose" in manifests
    manifest = manifests["cellpose"]
    assert isinstance(manifest, ModelManifest)
    assert manifest.name == "Cellpose"
    assert "cyto3" in manifest.versions
    assert "cyto" in manifest.versions["cyto3"].tasks


def test_flatten_manifest(temp_manifest_file):
    with open(temp_manifest_file) as f:
        raw_json = json.load(f)
    manifest = ModelManifest(**raw_json)
    flat = flatten_manifest(manifest)
    # Should have exactly one LocationEntry after flattening
    locations = flat.versions["cyto3"].tasks["cyto"].locations
    assert len(locations) == 1
    assert isinstance(locations[0].location, str)


def test_filter_location_and_empty_manifests(temp_manifest_file):
    # Load the manifest as raw JSON and instantiate ModelManifest
    with open(temp_manifest_file) as f:
        raw_json = json.load(f)
    manifest = ModelManifest(**raw_json)
    filtered, changed, num_removed = filter_location(manifest)
    # Only the first location for cyto3 is accessible (URL), cyto2 is not accessible (file path)
    assert changed is True
    assert num_removed == 1
    assert "cyto3" in filtered.versions
    assert (
        "cyto2" not in filtered.versions or len(filtered.versions["cyto2"].tasks) == 0
    )
    # Now test filter_empty_manifests
    manifests_dict = {filtered.short_name: filtered}
    filtered_dict = filter_empty_manifests(manifests_dict)
    assert filtered.short_name in filtered_dict


def test_filter_location_preserves_paired_config_path():
    # The accessible entry's config_path must be retained alongside its location.
    manifest_data = {
        "name": "PairedConfig",
        "short_name": "paired_config",
        "metadata": {"description": "Test paired config_path"},
        "versions": {
            "v1": {
                "tasks": {
                    "mito": {
                        "locations": [
                            {"location": "file:///nonexistent/path", "config_path": "file:///nonexistent/cfg1.yml"},
                            {"location": "https://example.com/model", "config_path": "/nonexistent/cfg2.yml"},
                        ]
                    }
                }
            }
        },
    }
    manifest = ModelManifest(**manifest_data)
    filtered, changed, num_removed = filter_location(manifest)
    assert changed is False
    task = filtered.versions["v1"].tasks["mito"]
    assert len(task.locations) == 1
    assert task.locations[0].location == "https://example.com/model"
    assert task.locations[0].config_path == "/nonexistent/cfg2.yml"


def test_empty_locations_raises():
    from pydantic import ValidationError
    manifest_data = {
        "name": "Bad Model",
        "short_name": "bad_model",
        "metadata": {"description": "Test"},
        "versions": {
            "v1": {"tasks": {"cyto": {"locations": []}}}
        },
    }
    with pytest.raises(ValidationError):
        ModelManifest(**manifest_data)


def test_filter_location_no_change(tmp_path):
    # Manifest with all accessible locations (URLs)
    manifest_data = {
        "name": "TestModel",
        "short_name": "testmodel",
        "metadata": {
            "description": "Test model with all accessible locations.",
        },
        "versions": {
            "v1": {
                "tasks": {
                    "cyto": {
                        "locations": [
                            {"location": "https://example.com/model1"},
                            {"location": "https://example.com/model2"},
                        ],
                    }
                }
            }
        },
        "params": [{"name": "Param1", "arg_name": "param1", "value": 1}],
    }
    manifest = ModelManifest(**manifest_data)
    filtered, changed, num_removed = filter_location(manifest)
    assert changed is False
    assert num_removed == 0
    # The task should still exist
    assert "cyto" in filtered.versions["v1"].tasks


@pytest.mark.parametrize(
    "input_value,expected",
    [
        (None, False),
        ("https://example.com/model", True),
        ("file:///nonexistent/path", False),
        ("/nonexistent/path/to/model.pt", False),
    ],
)
def test_is_accessible_param(input_value, expected):
    assert is_accessible(input_value) is expected


def test_is_accessible_with_tempfile(tmp_path):
    # Create a real file using tmp_path (pytest fixture)
    real_file = tmp_path / "afile.txt"
    real_file.write_text("test")
    assert is_accessible(str(real_file))
    # Nonexistent file in tmp_path
    assert not is_accessible(str(tmp_path / "doesnotexist.txt"))


@pytest.mark.skipif(
    sys.platform == "win32" or os.getuid() == 0,
    reason="Cannot restrict permissions on Windows or as root",
)
def test_is_accessible_permission_denied(tmp_path):
    # Reproduce: file exists but os.stat raises EACCES (errno 13) because the
    # parent directory has its execute/search bit removed.
    # In Python 3.12+, Path.exists() only suppresses ENOENT/ENOTDIR/EBADF/ELOOP;
    # EACCES propagates, so is_accessible raises PermissionError instead of
    # returning False.
    subdir = tmp_path / "restricted"
    subdir.mkdir()
    model_file = subdir / "model.pt"
    model_file.write_text("fake model weights")

    # Confirm the file is accessible before we restrict it
    assert is_accessible(str(model_file))

    # Remove execute (search) bit from the parent directory so that any
    # os.stat() on a path inside it raises PermissionError (errno 13)
    subdir.chmod(0o666)
    try:
        # Bug: PermissionError propagates out of is_accessible instead of
        # being caught and returning False
        result = is_accessible(str(model_file))
        assert result is False
    finally:
        # Restore permissions so that pytest's tmp_path cleanup can delete the dir
        subdir.chmod(0o755)


class TestModelParamDefault:
    def test_list_no_default_uses_first(self):
        """Without `default`, the first list item determines dtype and is the implicit default."""
        p = ModelParam(name="mode", value=["fast", "slow", "accurate"])
        assert p.default is None
        assert p._dtype is str

    def test_list_default_non_first_item(self):
        """Setting `default` to a non-first list item is accepted and reflected in _dtype."""
        p = ModelParam(name="mode", value=["fast", "slow", "accurate"], default="accurate")
        assert p.default == "accurate"
        assert p._dtype is str

    def test_list_default_int(self):
        """Integer default picks the correct dtype."""
        p = ModelParam(name="level", value=[1, 2, 3], default=3)
        assert p.default == 3
        assert p._dtype is int

    def test_default_not_in_list_raises(self):
        """A `default` value that is not in the choices list must raise a ValidationError."""
        with pytest.raises(ValidationError, match="not in the choices list"):
            ModelParam(name="mode", value=["fast", "slow"], default="medium")

    def test_default_on_scalar_raises(self):
        """`default` is only valid for list values; a scalar value must raise a ValidationError."""
        with pytest.raises(ValidationError, match="only be set when `value` is a list"):
            ModelParam(name="thresh", value=0.5, default=0.5)


# ---------------------------------------------------------------------------
# generate_default_config
# ---------------------------------------------------------------------------

MANIFEST_WITH_LIST_PARAMS = {
    "name": "Test List Model",
    "short_name": "test_list",
    "metadata": {"description": "Test"},
    "versions": {
        "v1": {
            "tasks": {
                "cyto": {"locations": [{"location": "https://example.com/model"}]}
            }
        }
    },
    "params": [
        {
            "name": "Plane",
            "arg_name": "plane",
            "value": ["XY", "XZ", "YZ", "All"],
        },
        {
            "name": "Median Filter Size",
            "arg_name": "median_slices",
            "value": [1, 3, 5, 7, 9, 11],
            "default": 3,
        },
    ],
}

MANIFEST_NO_PARAMS = {
    "name": "No Params Model",
    "short_name": "no_params",
    "metadata": {"description": "Model with no params"},
    "versions": {
        "v1": {
            "tasks": {
                "mito": {"locations": [{"location": "https://example.com/model"}]}
            }
        }
    },
}


def test_generate_default_config_scalar_params():
    manifest = ModelManifest(**EXAMPLE_MANIFEST)
    config_str = generate_default_config(manifest, "cyto3", "cyto")
    config = yaml.safe_load(config_str)
    assert isinstance(config, dict)
    assert "diameter" in config
    assert config["diameter"] == 0


def test_generate_default_config_list_param_with_explicit_default():
    manifest = ModelManifest(**MANIFEST_WITH_LIST_PARAMS)
    config_str = generate_default_config(manifest, "v1", "cyto")
    config = yaml.safe_load(config_str)
    # default=3 is set, so it should be used even though 1 is first in the list
    assert config["median_slices"] == 3


def test_generate_default_config_list_param_no_default_uses_first():
    manifest = ModelManifest(**MANIFEST_WITH_LIST_PARAMS)
    config_str = generate_default_config(manifest, "v1", "cyto")
    config = yaml.safe_load(config_str)
    # no default set, so first list element should be used
    assert config["plane"] == "XY"


def test_generate_default_config_no_params_returns_empty():
    manifest = ModelManifest(**MANIFEST_NO_PARAMS)
    config_str = generate_default_config(manifest, "v1", "mito")
    config = yaml.safe_load(config_str)
    assert config is None or config == {}


def test_generate_default_config_invalid_version_raises():
    manifest = ModelManifest(**EXAMPLE_MANIFEST)
    with pytest.raises(KeyError, match="nonexistent_version"):
        generate_default_config(manifest, "nonexistent_version", "cyto")


def test_generate_default_config_invalid_task_raises():
    manifest = ModelManifest(**EXAMPLE_MANIFEST)
    with pytest.raises(KeyError, match="nonexistent_task"):
        generate_default_config(manifest, "cyto3", "nonexistent_task")


# ---------------------------------------------------------------------------
# save_all_default_configs
# ---------------------------------------------------------------------------


def test_save_all_default_configs_creates_files(tmp_path):
    save_all_default_configs(output_dir=tmp_path)
    yaml_files = list(tmp_path.glob("*.yaml"))
    assert len(yaml_files) > 0


def test_save_all_default_configs_skips_no_params(tmp_path):
    save_all_default_configs(output_dir=tmp_path)
    yaml_files = list(tmp_path.glob("*.yaml"))
    # seai_unet defines no params - no file should be written for it
    seai_files = [f for f in yaml_files if "seai_unet" in f.name]
    assert len(seai_files) == 0


def test_save_all_default_configs_model_level_single_file(tmp_path):
    # Cellpose has model-level params only: all versions/tasks inherit them.
    # Exactly one file should be written: cellpose.yaml
    save_all_default_configs(output_dir=tmp_path)
    assert (tmp_path / "cellpose.yaml").exists()
    # No per-version/task files for cellpose
    task_files = [f for f in tmp_path.glob("cellpose_*.yaml")]
    assert len(task_files) == 0


def test_save_all_default_configs_valid_yaml_with_header(tmp_path):
    save_all_default_configs(output_dir=tmp_path)
    config_file = tmp_path / "cellpose.yaml"
    assert config_file.exists()
    content = config_file.read_text()
    assert content.startswith("# Auto-generated")
    # Strip header comment before parsing
    config = yaml.safe_load("\n".join(line for line in content.splitlines() if not line.startswith("#")))
    assert isinstance(config, dict) and len(config) > 0


def test_params_inherited_flag_model_level():
    # When a task has no params of its own, _params_inherited should be True
    manifest = ModelManifest(**EXAMPLE_MANIFEST)
    task = manifest.versions["cyto3"].tasks["cyto"]
    assert task._params_inherited is True


def test_params_inherited_flag_no_model_params_no_task_params():
    # When neither the manifest nor the task defines params, _params_inherited
    # must remain False - there is nothing to inherit.
    manifest = ModelManifest(**MANIFEST_NO_PARAMS)
    task = manifest.versions["v1"].tasks["mito"]
    assert task.params is None
    assert task._params_inherited is False


def test_short_name_auto_derived_from_name():
    """If short_name is omitted, the validator derives it from name."""
    manifest_data = {
        "name": "My Cool Model",
        "metadata": {"description": "Test"},
        "versions": {
            "v1": {"tasks": {"cyto": {"locations": [{"location": "https://example.com/model"}]}}}
        },
    }
    manifest = ModelManifest(**manifest_data)
    assert manifest.short_name == "my_cool_model"


def test_params_inherited_flag_task_level():
    # When a task defines its own params, _params_inherited should remain False
    manifest_data = {
        "name": "Task Params Model",
        "short_name": "task_params",
        "metadata": {"description": "Test"},
        "versions": {
            "v1": {
                "tasks": {
                    "cyto": {
                        "locations": [{"location": "https://example.com/model"}],
                        "params": [{"name": "Threshold", "arg_name": "threshold", "value": 0.5}],
                    }
                }
            }
        },
    }
    manifest = ModelManifest(**manifest_data)
    task = manifest.versions["v1"].tasks["cyto"]
    assert task._params_inherited is False


def test_save_all_default_configs_task_specific_file(tmp_path):
    # A manifest with no model-level params but task-specific params should
    # write a per-task file only - no model-level {short_name}.yaml
    manifest_data = {
        "name": "Task Only Model",
        "short_name": "task_only",
        "metadata": {"description": "Test"},
        "versions": {
            "v1": {
                "tasks": {
                    "cyto": {
                        "locations": [{"location": "https://example.com/model"}],
                        "params": [{"name": "Threshold", "arg_name": "threshold", "value": 0.5}],
                    }
                }
            }
        },
    }
    manifest_file = tmp_path / "task_only.json"
    manifest_file.write_text(json.dumps(manifest_data))
    out = tmp_path / "configs"
    save_all_default_configs(output_dir=out, paths=[manifest_file])
    assert (out / "task_only_v1_cyto.yaml").exists()
    assert not (out / "task_only.yaml").exists()
