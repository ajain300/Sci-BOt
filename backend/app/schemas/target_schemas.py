from pydantic import BaseModel, Field
from typing import Optional

class TargetBase(BaseModel):
    name: str
    weight: float
    scaling: Optional[str] = None
    unit: Optional[str] = None

class UncertaintyTargetConfig(TargetBase):
    type: str = "uncertainty"
    direction: str = Field(..., pattern="^(MIN|MAX)$")

class ExtremeTargetConfig(TargetBase):
    type: str = "extreme"
    direction: str = Field(..., pattern="^(MIN|MAX)$")

class ValueTargetConfig(TargetBase):
    type: str = "target"
    direction: str = "TARGET"
    target_value: float

class RangeTargetConfig(TargetBase):
    type: str = "range"
    direction: str = "RANGE"
    range_min: float
    range_max: float