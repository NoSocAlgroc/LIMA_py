from LIMA.lattice.LatticeState import LatticeState
from LIMA.scheduler.Scheduler import Scheduler
from LIMA.scheduler.Schedule import (
    Schedule,
    adjacentNodes,
    nodesAboveThreshold,
    buildTree,
    invertTree,
)


class LIMAScheduler(Scheduler):

    def schedule(self, state: LatticeState, threshold: float) -> Schedule:
        """Build this round's sampling tree from the nodes adjacent to
        `state.candidateEdges` whose nodeGrad exceeds `threshold`."""
        nodes = adjacentNodes(state.candidateEdges)
        nodes = nodesAboveThreshold(nodes, state, threshold)
        tree = buildTree(nodes, state)
        children = invertTree(tree, state)
        return Schedule(tree, children)
