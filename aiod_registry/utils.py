import json
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

import yaml

from aiod_registry import ModelManifest


def get_manifest_paths() -> list[Path]:
    json_dir = Path(__file__).parent.parent / "aiod_registry" / "manifests"
    return list(json_dir.glob("*.json"))


def is_accessible(location: str | None) -> bool:
    if location is None:
        return False
    res = urlparse(location)
    if res.scheme in ("file", ""):
        try:
            return Path(res.path).exists()
        except PermissionError:
            return False
    else:
        return True


def resolve_version(manifest: ModelManifest, version_input: str) -> str:
    """Return the registry key for a version given its exact name or slug.

    Tries an exact key match first, then falls back to slug matching.
    Raises KeyError with a helpful message if neither matches.
    """
    if version_input in manifest.versions:
        return version_input
    for key, version in manifest.versions.items():
        if version.slug == version_input:
            return key
    available = {k: v.slug for k, v in manifest.versions.items()}
    raise KeyError(
        f"Version '{version_input}' not found in manifest '{manifest.name}'. "
        f"Available versions (name: slug): {available}"
    )


def flatten_manifest(manifest: ModelManifest) -> ModelManifest:
    """
    Flatten the manifest by keeping only the first location entry.
    """
    # Make a deep copy of the manifest
    new_manifest = manifest.model_copy(deep=True)
    # Keep only the first (location, config_path) pair for each task
    for v_name, version in manifest.versions.items():
        for task_name, task in version.tasks.items():
            new_manifest.versions[v_name].tasks[task_name].locations = [task.locations[0]]
    return new_manifest


def filter_location(manifest: ModelManifest) -> tuple[ModelManifest, bool, int]:
    """
    Filter the locations list to the first accessible (location, config_path) pair.
    If no entry is accessible, remove the task entirely.
    """
    num = 0
    changed = False
    # Make a deep copy of the manifest
    new_manifest = manifest.model_copy(deep=True)
    # Loop through the versions and tasks and remove inaccessible ones
    for v_name, version in manifest.versions.items():
        for task_name, task in version.tasks.items():
            for entry in task.locations:
                if is_accessible(entry.location):
                    new_manifest.versions[v_name].tasks[task_name].locations = [entry]
                    break
            # If no location is accessible, remove the task completely
            else:
                del new_manifest.versions[v_name].tasks[task_name]
                changed = True
                num += 1
    return new_manifest, changed, num


def filter_empty_manifests(
    manifests: dict[str, ModelManifest],
) -> dict[str, ModelManifest]:
    # Track whether the whole manifest is empty
    remove = []
    for manifest in manifests.values():
        # Only keep versions that have a task remaining
        manifest.versions = {
            k: v for k, v in manifest.versions.items() if len(v.tasks) > 0
        }
        # If there are no versions, remove the manifest
        if len(manifest.versions) == 0:
            remove.append(True)
        else:
            remove.append(False)
    # Remove the empty manifests
    return {
        manifest.short_name: manifest
        for manifest, remove in zip(manifests.values(), remove)
        if not remove
    }


def load_manifests(
    paths: Optional[list[Union[Path, str]]] = None,
    filter_access: bool = False,
) -> dict[str, ModelManifest]:
    if paths is None:
        paths = get_manifest_paths()
    manifests = {}
    for path in paths:
        with open(path, "r") as f:
            json_manifest = json.load(f)
            manifest = ModelManifest(**json_manifest)
            manifests[manifest.short_name] = manifest
    # Remove those model versions that are not accessible (if a path is provided)
    if filter_access:
        # Track how many versions are removed
        num_versions_removed = 0
        # Dict to store the new manifests
        new_manifests = {}
        # Check that something has been changed, to allow for early return
        changed = False
        for manifest in manifests.values():
            new_manifest, changed_i, num = filter_location(manifest)
            # Needed now as filtering is encapsulated in a function
            if changed_i:
                changed = True
            num_versions_removed += num
            new_manifests[new_manifest.short_name] = new_manifest
        # Check how much of each manifest remains and prune if necessary
        if changed:
            # Print the number of versions removed
            print(f"Removed {num_versions_removed} inaccessible version(s)!")
            new_manifests = filter_empty_manifests(new_manifests)
            if len(new_manifests) != len(manifests):
                print(
                    f"Removed {len(manifests) - len(new_manifests)} empty manifest(s)!"
                )
            return new_manifests
        else:
            return new_manifests
    else:
        return manifests


def _params_to_yaml(params: list) -> str:
    """Serialise a list of ModelParam to a YAML string keyed by arg_name."""
    defaults = {}
    for param in params:
        if isinstance(param.value, list):
            defaults[param.arg_name] = (
                param.default if param.default is not None else param.value[0]
            )
        else:
            defaults[param.arg_name] = param.value
    return yaml.safe_dump(
        defaults, sort_keys=False, default_flow_style=False, allow_unicode=True
    )


def generate_default_config(manifest: ModelManifest, version: str, task: str) -> str:
    """Return a YAML string of default parameter values for a given model version and task.

    For list-valued params, the effective default is `param.default` if set, otherwise
    the first element. Returns an empty YAML mapping if the task has no params.

    Raises KeyError if the version or task is not found in the manifest.
    """
    if version not in manifest.versions:
        raise KeyError(
            f"Version '{version}' not found in manifest '{manifest.name}'. "
            f"Available versions: {list(manifest.versions.keys())}"
        )
    version_obj = manifest.versions[version]
    if task not in version_obj.tasks:
        raise KeyError(
            f"Task '{task}' not found in version '{version}' of manifest '{manifest.name}'. "
            f"Available tasks: {list(version_obj.tasks.keys())}"
        )
    task_obj = version_obj.tasks[task]
    if not task_obj.params:
        return yaml.dump({}, default_flow_style=False)
    return _params_to_yaml(task_obj.params)


def save_all_default_configs(
    output_dir: Union[Path, str] = "default_configs",
    paths: Optional[list[Union[Path, str]]] = None,
) -> None:
    """Generate and save default parameter config YAML files for all models.

    Saves one shared ``{short_name}.yaml`` for model-level params, plus a
    task-specific ``{short_name}_{version}_{task}.yaml`` only for tasks that
    define their own params (i.e. not inherited from the model level).
    Tasks and models with no params at all are skipped.

    ``paths`` is passed directly to :func:`load_manifests`; if omitted, all
    manifests in the registry are used.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    header = "# Auto-generated by save_all_default_configs - do not edit manually\n"
    manifests = load_manifests(paths=paths)
    for manifest in manifests.values():
        # Write one shared config for model-level params
        if manifest.params:
            config_str = _params_to_yaml(manifest.params)
            filepath = output_dir / f"{manifest.short_name}.yaml"
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(header)
                f.write(config_str)
            print(f"Saved {filepath}")

        # Write task-specific configs only where the task has its own params
        for version_name, version in manifest.versions.items():
            for task_name, task_obj in version.tasks.items():
                if task_obj._params_inherited:
                    continue  # Already covered by the model-level config
                if not task_obj.params:
                    print(
                        f"Skipping {manifest.short_name}/{version_name}/{task_name} — no params defined."
                    )
                    continue
                config_str = _params_to_yaml(task_obj.params)
                safe_version = version_name.replace(" ", "_")
                filename = f"{manifest.short_name}_{safe_version}_{task_name}.yaml"
                filepath = output_dir / filename
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(header)
                    f.write(config_str)
                print(f"Saved {filepath}")


def _gen_configs_cli() -> None:
    """Console script entry point for ``aiod-gen-configs``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="aiod-gen-configs",
        description="Generate default parameter config YAML files for all registered models.",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="default_configs",
        help="Directory to write configs into (default: ./default_configs).",
    )
    args = parser.parse_args()
    save_all_default_configs(output_dir=args.output_dir)
