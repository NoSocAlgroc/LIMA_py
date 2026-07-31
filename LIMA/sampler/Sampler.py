from abc import ABC, abstractmethod
from typing import List

from data.domain.Domain import Domain
from data.predicate.PredicateGroup import PredicateGroup
from LIMA.lattice.LatticeState import LatticeState
from LIMA.sampler.Sample import Sample
from LIMA.scheduler.Schedule import Schedule


class Sampler(ABC):
    """Abstract: draws n samples from a domain following a schedule."""

    @abstractmethod
    def sample(
        self,
        sched: Schedule,
        domain: Domain,
        predicate_groups: List[PredicateGroup],
        state: LatticeState,
        n: int,
    ) -> Sample:
        raise NotImplementedError
