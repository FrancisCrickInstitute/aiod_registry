import builtins
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    model_validator,
)

TASK_NAMES = {
    "mito": "Mitochondria",
    "er": "Endoplasmic Reticulum",
    "ne": "Nuclear Envelope",
    "everything": "Everything!",
    "nuclei": "Nuclei",
    "cyto": "Cytoplasm",
    "drop": "Lipid Droplets",
    "boundaries": "Boundaries",
}
task_names = "|".join(TASK_NAMES.keys())

# Define custom types/fields
# Centralise to make it easier to change later
# Regex pattern to match task names, ignoring case
Task = Annotated[str, Field(..., pattern=rf"^(?i:{task_names})$")]
ModelName = Annotated[str, Field(..., min_length=1, max_length=50)]
ParamName = Annotated[
    str,
    Field(
        ...,
        min_length=1,
        max_length=50,
        description="Name of the parameter. If `arg_name` is not provided, this will be used as the argument name to the underlying model.",
    ),
]
ParamValue = Annotated[
    str | int | float | bool | None | list[str | int | float | bool],
    Field(
        ...,
        description="Default parameter value. If a list, the parameters will be treated as dropdown choices. Use the `default` field on ModelParam to specify which item is selected by default (otherwise the first item is used). The type of the default (or first) element determines the parameter type.",
    ),
]
Usage = Annotated[
    str | Path | AnyUrl,
    Field(
        ...,
        title="Usage Guide",
        description="A path to a file, a URL, or a string containing the usage guide for the model.",
    ),
]


def _validate_axes(v: str) -> str:
    if len(set(v)) != len(v):
        raise ValueError("Axes must not contain repeated characters.")
    if "Y" not in v or "X" not in v:
        raise ValueError("Axes must contain at least Y and X.")
    return v


Axes = Annotated[
    str,
    Field(
        ...,
        min_length=2,
        pattern=r"^[TCZYX]+$",
        description="Axes specification for the model (e.g., 'YX' for 2D, 'ZYX' for 3D, 'CZYX' with channels). Must contain at least Y and X, with no repeated letters. Valid characters: T, C, Z, Y, X.",
    ),
    AfterValidator(_validate_axes),
]


def print_attr(attr, br: bool = True):
    "Shorthand to print something in brackets or not, only if not None."
    if attr is None:
        return ""
    if br:
        return f"({attr})"
    else:
        return f"{attr}"


def shorten_name(name: str) -> str:
    return "_".join(name.lower().split(" "))


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ChannelStart = Annotated[
    Literal[-1, 0],
    Field(
        description=(
            "Lowest integer value shown in the channel dropdown. "
            "-1 means the first item is 'original image as-is' and image channels are 0-based. "
            "0 means the first item is model-specific and image channels are 1-based."
        ),
    ),
]


class ModelParam(StrictModel):
    name: ParamName
    arg_name: str | None = None
    value: ParamValue
    default: str | int | float | bool | None = None  # Override default for list values
    tooltip: str | None = None
    dtype: str | None = None  # Used if default value is None
    param_type: str | None = None  # e.g. "channel" for image-aware channel selectors
    channel_start: ChannelStart = -1
    channel_start_label: Annotated[
        str,
        Field(
            min_length=1,
            description="Label for the first channel dropdown item.",
            examples=["original", "Grayscale", "No nucleus channel"],
        ),
    ] = "original"
    _dtype = None  # Determined from value if given

    @model_validator(mode="after")
    def create_arg_name(self):
        if self.arg_name is None:
            self.arg_name = self.name
        return self

    @model_validator(mode="after")
    def validate_default(self):
        if self.default is not None:
            if not isinstance(self.value, list):
                raise ValueError(
                    f"Parameter {self.name}: `default` can only be set when `value` is a list."
                )
            if self.default not in self.value:
                raise ValueError(
                    f"Parameter {self.name}: `default` value {self.default!r} is not in the choices list {self.value}."
                )
        return self

    @model_validator(mode="after")
    def extract_arg_type(self):
        if isinstance(self.value, list):
            reference = self.default if self.default is not None else self.value[0]
            self._dtype = type(reference)
        else:
            self._dtype = type(self.value)
        # If None, we need a dtype to poss cast to when dealing with GUIs
        if self.value is None:
            if self.dtype is None:
                raise ValueError(
                    f"Parameter {self.name} needs a dtype if default value is None!"
                )
            else:
                if getattr(builtins, self.dtype, None) is None:
                    raise ValueError(
                        f"Parameter {self.name} has an invalid dtype ({self.dtype})!"
                    )
        else:
            self.dtype = self._dtype
        return self

    @model_validator(mode="after")
    def validate_channel_config(self):
        if self.param_type != "channel":  # noqa: SIM102
            if self.channel_start != -1 or self.channel_start_label != "original":
                raise ValueError(
                    "`channel_start` and `channel_start_label` can only be customized "
                    'when `type` is "channel".'
                )
        return self


class Author(StrictModel):
    name: str
    affiliation: str
    email: str | None = None
    url: AnyUrl | None = None
    github: str | None = None
    orcid: str | None = None


class Publication(StrictModel):
    title: str
    info: Annotated[
        str,
        Field(
            ...,
            description="Information on publication, whether it pertains to the model or the underlying data or something else.",
        ),
    ]
    url: AnyUrl
    year: int | None = None
    doi: str | None = None
    authors: list[Author] | None = None


class Metadata(StrictModel):
    description: Annotated[
        str,
        Field(
            ...,
            description="A short description of the model to provide context.",
        ),
    ]
    authors: list[Author] | None = None
    pubs: list[Publication] | None = None
    url: AnyUrl | None = None
    repo: AnyUrl | None = None

    def __str__(self):
        misc_info = (
            f"{'URL: ' + print_attr(self.url, br=False) if self.url is not None else ''}\n"
            f"{'Repo: ' + print_attr(self.repo, br=False) if self.repo is not None else ''}\n"
        )

        if self.pubs is None:
            all_pubs = ""
        else:
            all_pubs = "\nPublications:\n" + "\n-".join(
                [
                    (
                        f"{pub.title} {print_attr(pub.year)}- {pub.url}"
                        f"{', DOI: ' + print_attr(pub.doi, br=False) if pub.doi is not None else ''}\n"
                    )
                    for pub in self.pubs
                ]
            )
        return f"Description: {self.description}\n{misc_info if len(misc_info) > 0 else ''}{all_pubs}"


class LocationEntry(StrictModel):
    location: str = Field(..., description="A URL or file path to the model artifact.")
    config_path: str | None = Field(
        None,
        description="Optional path or URL to the config file paired with this location.",
    )


class ModelVersionTask(StrictModel):
    locations: list[LocationEntry] = Field(
        ...,
        description="Ordered list of (location, config_path) pairs. The first accessible entry is used.",
        min_length=1,
    )
    params: list[ModelParam] | None = None
    metadata: Metadata | None = None
    _params_inherited: bool = PrivateAttr(default=False)


class ModelVersion(StrictModel):
    axes: Axes | None = None
    tasks: dict[Task, ModelVersionTask]
    metadata: Metadata | None = None
    slug: str = Field(
        default="",
        description="Filesystem-safe identifier derived from the version name (lowercase, spaces replaced with underscores). Auto-derived if not set.",
    )


class ModelManifest(StrictModel):
    name: str = Field(..., min_length=1, max_length=50)
    short_name: str = ""
    versions: dict[ModelName, ModelVersion]
    params: list[ModelParam] | None = None
    config: Path | None = None
    metadata: Metadata
    usage_guide: Usage | None = None

    @model_validator(mode="after")
    def create_short_name(self):
        if not self.short_name:
            self.short_name = shorten_name(self.name)
        return self

    # Embed base model params into each version if not provided
    @model_validator(mode="after")
    def fill_empty_params(self):
        for version in self.versions.values():
            for task in version.tasks.values():
                if task.params is None and self.params is not None:
                    task.params = self.params
                    task._params_inherited = True
        return self

    @model_validator(mode="after")
    def fill_version_slugs(self):
        for version_name, version in self.versions.items():
            if not version.slug:
                version.slug = shorten_name(version_name)
        return self


if __name__ == "__main__":
    import json

    schema_fpath = Path(__file__).parent / "schema.json"

    # Write the schema to file
    with open(schema_fpath, "w") as f:
        f.write(json.dumps(ModelManifest.model_json_schema(), indent=2))
