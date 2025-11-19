from dataclasses import dataclass, field
from typing import Any


@dataclass
class InferenceResult:
    """Class for storing inference results."""

    time: dict[str, Any] = field(
        default_factory=lambda: {"total": 0.0, "sections": {}},
        kw_only=True,  # Makes this field keyword-only, so it doesn't conflict with positional args in subclasses
    )

    fields: dict[str, Any] = field(default_factory=dict, kw_only=True)

    def add_time(self, section_name: str, value: float) -> None:
        """Add a time measurement to the sections dictionary and update total time.

        Args:
            section_name: Name of the section (e.g., "model_loading", "inference")
            value: Time in seconds for this section
        """
        self.time["sections"][section_name] = value
        self.time["total"] = sum(self.time["sections"].values())

    def get_total_time(self) -> float:
        """Get the total time from all sections."""
        return self.time["total"]

    def get_sections(self) -> dict[str, float]:
        """Get the sections time dictionary."""
        return self.time["sections"]

    def add_field(self, field_name: str, value: Any) -> None:
        """Add a field to the result."""
        self.fields[field_name] = value

    def __getattr__(self, name: str) -> Any:
        """Allow accessing fields as attributes."""
        if name in self.fields:
            return self.fields[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Allow setting fields as attributes if they're not dataclass fields."""
        # Check if it's a dataclass field
        if (
            hasattr(self.__class__, "__dataclass_fields__")
            and name in self.__class__.__dataclass_fields__
        ):
            super().__setattr__(name, value)
        else:
            # It's a dynamic field
            if not hasattr(self, "fields"):
                super().__setattr__("fields", {})
            self.fields[name] = value

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a dictionary."""
        result = {
            "time": self.time,  # Keep the full nested time structure
        }
        # Add all dynamic fields
        result.update(self.fields)
        return result
