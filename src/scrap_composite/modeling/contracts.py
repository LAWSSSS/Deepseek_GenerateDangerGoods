"""Inputs offered by both the CLI and modeling workbench."""
from typing import Literal

from pydantic import Field

from scrap_composite.contracts import Strict

View = Literal['front', 'back', 'left', 'right', 'top', 'bottom']
Operation = Literal['rebuild', 'retexture', 'verify']


class BuildOptions(Strict):
    operation: Operation = 'rebuild'
    subject: str | None = Field(default=None, max_length=250)
    major_mm: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    dimension_source: str | None = Field(default=None, max_length=500)
    material_library: str | None = None


class Publication(Strict):
    version: str
