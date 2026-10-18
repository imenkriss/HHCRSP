"""Typed domain objects for the multi-time-window optimization model.

Benchmark parsing does not supply clinical demand or caregiver skill categories;
those fields use explicit placeholder defaults until an instance enrichment step
provides them.
"""

from dataclasses import dataclass, field
import math


@dataclass
class Patient:
    id: str | int
    x: float
    y: float
    service_time: float
    demand_level: int = 1
    time_windows: list[tuple[float, float]] = field(default_factory=list)
    preferences: dict[str | int, float] = field(default_factory=dict)
    rejections: dict[str | int, bool] = field(default_factory=dict)
    sampled_window_widths: list[float] = field(default_factory=list)
    clipped_window_widths: list[bool] = field(default_factory=list)
    requested_window_count: int | None = None
    used_window_count: int | None = None
    # Non-clinical attributes carried from the teacher CSV when patients are
    # supplied by data.loader (ASSUMPTION documented in docs/ASSUMPTIONS.md).
    care_type: str | None = None
    priority: str = "Medium"
    # sigma_i: relative uncertainty level of the skill demand (P5).
    demand_uncertainty: float = 0.0


@dataclass(frozen=True)
class ArrivalTiming:
    """Soft-window cost components for one patient arrival, in minutes."""

    waiting_time: float = 0.0
    tardiness: float = 0.0


def evaluate_window_arrival(arrival_time: float, window: tuple[float, float]) -> ArrivalTiming:
    """Return soft-window early waiting and late tardiness separately.

    Neither component is a hard feasibility failure. Early arrivals wait until
    the window opens; late arrivals incur tardiness after its latest bound.
    """
    earliest, latest = window
    return ArrivalTiming(
        waiting_time=max(0.0, earliest - arrival_time),
        tardiness=max(0.0, arrival_time - latest),
    )


@dataclass
class Caregiver:
    id: str | int
    skill_level: int = 1
    start_time: float = 0
    rated_minutes: int = 480
    max_overtime_minutes: int = 240
    capacity: float | None = None
    specialty: str | None = None
    # beta_k: relative uncertainty level of the departure time (P5).
    departure_uncertainty: float = 0.0

    @property
    def hard_daily_limit_minutes(self) -> int:
        """Rated working time plus the allowed overtime cap."""
        return self.rated_minutes + self.max_overtime_minutes


@dataclass
class Instance:
    depot: dict[str, float]
    laboratory: dict[str, float] | None
    caregivers: list[Caregiver]
    patients: list[Patient]
    horizon: tuple[float, float]
    travel_speed: float | None = None
    vehicle_capacity: float | None = None
    name: str = ""
    max_patients_per_caregiver: int | None = None
    seed: int | None = None
    # alpha_ij: relative uncertainty level per ordered arc (P5).
    travel_uncertainty: dict[tuple, float] = field(default_factory=dict)

    def _node_coordinates(self, node_id: str | int) -> tuple[float, float]:
        if node_id == "depot":
            return float(self.depot["x"]), float(self.depot["y"])
        if node_id == "laboratory":
            if self.laboratory is None:
                raise KeyError(node_id)
            return float(self.laboratory["x"]), float(self.laboratory["y"])
        for patient in self.patients:
            if patient.id == node_id:
                return patient.x, patient.y
        raise KeyError(node_id)

    def distance(self, a: str | int, b: str | int) -> float:
        """Return Euclidean distance between the depot, lab, or patient nodes."""
        return math.dist(self._node_coordinates(a), self._node_coordinates(b))

    def travel_time(self, a: str | int, b: str | int) -> float:
        """Return travel minutes between nodes using this instance's speed."""
        if self.travel_speed is None or self.travel_speed <= 0:
            raise ValueError("travel_speed must be a positive number")
        return self.distance(a, b) / self.travel_speed
