import json

import pytest
from pydantic import ValidationError

from aiod_registry import ModelManifest, get_manifest_paths


@pytest.mark.parametrize("json_path", get_manifest_paths(), ids=lambda x: x.name)
def test_manifest(json_path):
    with json_path.open("r") as f:
        json_manifest = json.load(f)
        try:
            ModelManifest.model_validate(json_manifest)
        except ValidationError as e:
            raise e
