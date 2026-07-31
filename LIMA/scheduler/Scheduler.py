from abc import ABC, abstractmethod

from LIMA.lattice.LatticeState import LatticeState
from LIMA.scheduler.Schedule import Schedule


class Scheduler(ABC):
    """Abstract: decides this round's sampling tree from the lattice state."""

    @abstractmethod
    def schedule(self, state: LatticeState, threshold: float) -> Schedule:
        raise NotImplementedError
