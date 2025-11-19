from dataclasses import dataclass, field
from typing import Any


@dataclass
class InferenceResult:
    """Base class for inference results.

    All inference functions should return a subclass of this class.
    """

    time: dict[str, Any] = field(
        default_factory=lambda: {"time": 0.0, "sections": {}}
    )

    def add_time(self, section_name: str, value: float) -> None:
        """Add a time measurement to the sections dictionary and update total time.

        Args:
            section_name: Name of the section (e.g., "model_loading", "inference")
            value: Time in seconds for this section
        """
        self.time["sections"][section_name] = value
        self.time["time"] = sum(self.time["sections"].values())

    def get_total_time(self) -> float:
        """Get the total time from all sections."""
        return self.time["time"]

    def get_sections(self) -> dict[str, float]:
        """Get the sections time dictionary."""
        return self.time["sections"]
