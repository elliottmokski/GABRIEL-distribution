"""Pipeline primitives."""
from typing import Iterable, Any


class Pipeline:
    """Very small placeholder for a processing pipeline."""

    def __init__(self, steps: Iterable[Any]):
        self.steps = list(steps)

    def run(self, data: Any) -> Any:
        """Run all steps sequentially."""
        for step in self.steps:
            if hasattr(step, "fit"):
                step.fit(data)
            if hasattr(step, "transform"):
                data = step.transform(data)
        return data
