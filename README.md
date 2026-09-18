# AIoD Model Registry

[![PyPI](https://img.shields.io/pypi/v/aiod-registry.svg)](https://pypi.org/project/aiod-registry/)
[![Python versions](https://img.shields.io/pypi/pyversions/aiod-registry.svg)](https://pypi.org/project/aiod-registry/)
[![Tests](https://github.com/FrancisCrickInstitute/aiod_registry/actions/workflows/run_tests.yml/badge.svg)](https://github.com/FrancisCrickInstitute/aiod_registry/actions/workflows/run_tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-aiod__docs-1f6feb.svg)](https://franciscrickinstitute.github.io/aiod_docs/sections/model_registry/)

The central registry of models available within [AI OnDemand (AIoD)](https://franciscrickinstitute.github.io/aiod_docs). Each model family is described by a JSON *manifest* validated against a strict [Pydantic](https://docs.pydantic.dev/) schema, and that manifest is the single source of truth for the whole framework: it tells the [Napari plugin](https://github.com/FrancisCrickInstitute/aiod_napari) what to render (model name, parameters etc.), and the [Segment-Flow](https://github.com/FrancisCrickInstitute/Segment-Flow) pipeline where to fetch checkpoints and configs from.

Adding a model to AIoD usually means adding or editing a manifest here. No UI code, and sometimes no pipeline code, needs to change.

This package provides:

- The manifest schema itself, and tests that validate every manifest against it
- Utility functions for ingesting manifests and filtering by whether a user has access to each model, enabling us to [automatically build the Napari plugin's UI](https://franciscrickinstitute.github.io/aiod_docs/sections/development/#automatic-ui-construction)
- CLI entry points for generating default model configs (`aiod-gen-configs`) and the JSON schema (`aiod-gen-schema`)

## Requirements

Python 3.11 or 3.12.

## Installation

```bash
uv add aiod-registry  # or: uv pip install aiod-registry / pip install aiod-registry
```

Note that you should do this from within a conda, uv etc. environment.

## Quick start

```python
from aiod_registry import TASK_NAMES, load_manifests

manifests = load_manifests()  # {short_name: ModelManifest}, from the bundled manifests
cellpose = manifests["cellpose"]

print(cellpose.name, list(cellpose.versions))
print(TASK_NAMES)  # every segmentation task AIoD knows about

# Hide model versions this user has no filesystem access to
accessible = load_manifests(filter_access=True)
```

The set of models this resolves to is rendered in the docs: see [Available Models](https://franciscrickinstitute.github.io/aiod_docs/sections/model_registry/models/).

## Documentation

Full documentation for AIoD lives at **[franciscrickinstitute.github.io/aiod_docs](https://franciscrickinstitute.github.io/aiod_docs/)**.

| Topic | Link |
| --- | --- |
| The manifest schema, field by field | [Model Registry](https://franciscrickinstitute.github.io/aiod_docs/sections/model_registry/) |
| Every model currently available | [Available Models](https://franciscrickinstitute.github.io/aiod_docs/sections/model_registry/models/) |
| Adding a model, version, task or location | [Expanding AIoD](https://franciscrickinstitute.github.io/aiod_docs/sections/contributing/expanding/) |
| How model location controls access | [Concepts → Model location](https://franciscrickinstitute.github.io/aiod_docs/sections/concepts/#model-location) |
| Family vs version vs task | [AIoD Concepts](https://franciscrickinstitute.github.io/aiod_docs/sections/concepts/#model-family) |

## Contributing

Contributions are very welcome! To add a model, follow [Expanding AIoD](https://franciscrickinstitute.github.io/aiod_docs/sections/contributing/expanding/); for wider development across the AIoD repos, see the [AIoD Developer Guide](https://franciscrickinstitute.github.io/aiod_docs/sections/contributing/developing/).

### Development setup

```bash
git clone https://github.com/FrancisCrickInstitute/aiod_registry.git
cd aiod_registry
uv sync  # or: pip install -e ".[dev]"
```

### Local validation

To check whether a new manifest is valid, run the test suite — any problems are reported in detail by Pydantic:

```bash
uv run pytest -v tests/
```

These tests also run automatically on every pull request.

## Support

Please [open an issue](https://github.com/FrancisCrickInstitute/aiod_registry/issues) for bugs, or to request a model that isn't yet in the registry. See the docs [Support](https://franciscrickinstitute.github.io/aiod_docs/sections/support/) page for other ways to get in touch (cameron.shand@crick.ac.uk, jon.smith@crick.ac.uk).

## License

MIT — see [LICENSE](LICENSE).
