import math
from collections import defaultdict
from LIMA.lattice.Lattice import Lattice
from LIMA.lattice.LatticeState import LatticeState
from util.stats.betaDist import BetaDist, nodeDist4
from typing import Dict, Iterator, List, Optional, Set, Tuple


class LIMALatticeState(LatticeState):
    """Concrete LatticeState: stores a BetaDist per explored node and per
    sampled edge, plus a cache of sound/candidate edges maintained via a
    statistical soundness test as new sampling results come in.
    """


    def __init__(self, lattice: Lattice) -> None:
        self.lattice = lattice
        self._dists: Dict[Lattice.Node, BetaDist] = {}
        self._edge_dists: Dict[Lattice.Edge, BetaDist] = {}
        # Cached set of edges deemed sound by the statistical test.
        # Populated/updated externally
        self.soundEdges: Set[Lattice.Edge] = set()
        self.candidateEdges: Set[Lattice.Edge] = set()
        # Root always starts explored with uninformative prior
        self._dists[lattice.getRoot()] = BetaDist.empty()

        # Seed: every root -> depth-1 edge is sound, and every
        # depth-1 -> depth-2 edge is a candidate.
        root = lattice.getRoot()
        for pg in range(lattice.npgs):
            for p in range(lattice.pgs[pg]):
                edge = root.to(pg, p)
                self.soundEdges.add(edge)

                child = edge.to()
                for pg2 in range(lattice.npgs):
                    if pg2 == pg:
                        continue
                    for p2 in range(lattice.pgs[pg2]):
                        self.candidateEdges.add(child.to(pg2, p2))

    def getDist(self, n: Lattice.Node) -> BetaDist:
        return self._dists.get(n, BetaDist.empty())

    def getEdgeDist(self, edge: Lattice.Edge) -> BetaDist:
        return self._edge_dists.get(edge, BetaDist.empty())

    def isEdgeExplored(self, edge: Lattice.Edge) -> bool:
        return edge in self._edge_dists

    def exploredEdges(self) -> Iterator[Lattice.Edge]:
        return iter(self._edge_dists)

    def isExplored(self, n: Lattice.Node) -> bool:
        return n in self._dists

    def explored(self) -> Iterator[Lattice.Node]:
        return iter(self._dists)

    def updateNode(self, n: Lattice.Node, a: int, b: int) -> None:
        """Add a new batch of (successes, failures) to a node's distribution."""
        if n not in self._dists:
            self._dists[n] = BetaDist.empty()
        self._dists[n].add(a, b)

    def updateEdge(self, edge: Lattice.Edge, a: int, b: int) -> None:
        """Add a conditional observation (child successes, parent successes
        not child) to an edge's distribution."""
        if edge not in self._edge_dists:
            self._edge_dists[edge] = BetaDist.empty()
        self._edge_dists[edge].add(a, b)

    def update(
        self,
        results: Dict[Lattice.Node, Tuple[int, int]],
        threshold: float,
        edge_results: Optional[Dict[Lattice.Edge, Tuple[int, int]]] = None,
    ) -> None:
        """Add a sampling round's results, then refresh sound/candidate edges.

        With `edge_results`, conditional edge distributions are updated too
        and soundness is tested on them; without it, the legacy node-only
        path is used.
        """
        for n, (a, b) in results.items():
            self.updateNode(n, a, b)
        if edge_results is not None:
            for edge, (a, b) in edge_results.items():
                self.updateEdge(edge, a, b)
        self._updateSoundEdges(threshold, edge_results is not None)

    def _updateSoundEdges(self, threshold: float, use_edge_stats: bool = False) -> None:
        newly_sound: List[Lattice.Edge] = []
        for edge in list(self.candidateEdges):
            sound = (
                self._isSoundByEdge(edge, threshold)
                if use_edge_stats
                else self._isSoundByNode(edge, threshold)
            )
            if sound:
                self.candidateEdges.discard(edge)
                self.soundEdges.add(edge)
                newly_sound.append(edge)
        for edge in newly_sound:
            self.propagateSound(edge)

    def getTreeParent(self, n: Lattice.Node) -> Lattice.Node:
        """Spanning-tree parent of n: remove the predicate whose removal gives the
        most selective (lowest meanLO) parent node."""
        best_pg = min(n.preds, key=lambda pg: self.getDist(n.fr(pg).fr()).meanLO)
        return n.fr(best_pg).fr()

    def initializeToDepth(self, k: int) -> None:
        """Pre-populate nodes and sound edges for all lattice nodes up to depth k.

        Every node at depth ≤ k gets an uninformative BetaDist(1,1), and every
        upward edge between adjacent layers is added to soundEdges. This lets
        the scheduler immediately propose depth-(k+1) candidates on the first step,
        treating the shallower layers as already validated.
        """

        visited: Set[Lattice.Node] = {self.lattice.getRoot()}
        frontier: List[Lattice.Node] = [self.lattice.getRoot()]
        for _ in range(k):
            next_frontier: List[Lattice.Node] = []
            for node in frontier:
                for pg in range(self.lattice.npgs):
                    if pg in node:
                        continue
                    for p in range(self.lattice.pgs[pg]):
                        edge = node.to(pg, p)
                        child = edge.to()
                        self.soundEdges.add(edge)
                        if child not in visited:
                            visited.add(child)
                            self._dists[child] = BetaDist.empty()
                            next_frontier.append(child)
            frontier = next_frontier

    def getTreeParentEdge(self, n: Lattice.Node) -> Lattice.Edge:
        """Upward edge from tree-parent to n (parent → n, adding best_pg)."""
        best_pg = min(n.preds, key=lambda pg: self.getDist(n.fr(pg).fr()).meanLO)
        parent = n.fr(best_pg).fr()
        return parent.to(best_pg, n.preds[best_pg])

    def _supportEdge(self, edge: Lattice.Edge, pg: int) -> Lattice.Edge:
        """The same added predicate applied to the parent without `pg`."""
        parent = edge.fr()
        return parent.fr(pg).fr().to(edge.pg, edge.p)

    def _isSoundByNode(self, edge: Lattice.Edge, threshold: float) -> bool:
        """Legacy node-distribution soundness test.

        Edge parent→child is sound if, for every predicate pg in parent,
        nodeDist4(parent-pg, child-pg, parent, child) < -threshold.

        Each comparison asks: does this edge add more selectivity than the
        reference edge that lacks predicate pg?
        """
        parent = edge.fr()
        from1_dist = self.getDist(parent)
        to1_dist = self.getDist(edge.to())
        for pg in parent.preds:
            edge2 = self._supportEdge(edge, pg)
            z = nodeDist4(
                from1_dist,
                to1_dist,
                self.getDist(parent.fr(pg).fr()),
                self.getDist(edge2.to()),
            )
            if z >= -threshold:
                return False
        return True

    def _isSoundByEdge(self, edge: Lattice.Edge, threshold: float) -> bool:
        """Edge-distribution soundness test: compare the conditional edge
        against every reduced-parent support edge. Missing edge statistics
        fail closed.
        """
        if not self.isEdgeExplored(edge):
            return False

        for pg in edge.fr().preds:
            support = self._supportEdge(edge, pg)
            if not self.isEdgeExplored(support):
                return False
            if self._edgeInteractionZ(edge, support) >= -threshold:
                return False
        return True

    def _minSoundZ(self, edge: Lattice.Edge) -> float:
        """The minimum edge-interaction z-score over every predicate pg in
        the edge's parent - how sound `edge` is at its strongest supporting
        comparison, on the same scale `_isSoundByEdge` thresholds against.
        """
        if not self.isEdgeExplored(edge):
            return math.inf

        best = math.inf
        for pg in edge.fr().preds:
            support = self._supportEdge(edge, pg)
            if not self.isEdgeExplored(support):
                return math.inf
            best = min(best, self._edgeInteractionZ(edge, support))
        return best

    def _edgeInteractionZ(self, edge: Lattice.Edge, support: Lattice.Edge) -> float:
        """Difference in conditional log-odds: negative means the edge is
        more selective than its support edge.

        If the edge has too little tail evidence, its own log-odds estimate
        is driven mostly by its sample size. In that case the support is
        replaced locally (never stored) by a distribution with the edge's
        sample size and the support's rate, so both sides are compared at
        equal evidence.
        """
        e = self.getEdgeDist(edge)
        s = self.getEdgeDist(support)

        if e.varLogOdds > 0.5:
            n_edge = (e.a - 1) + (e.b - 1)
            n_support = (s.a - 1) + (s.b - 1)
            if 0 < n_edge <= n_support:
                raw_a = int(round(n_edge * s.mean))
                raw_a = min(max(raw_a, 0), n_edge)
                s = BetaDist.empty()
                s.add(raw_a, n_edge - raw_a)

        m = e.meanLogOdds - s.meanLogOdds
        v = e.varLogOdds + s.varLogOdds
        return m / math.sqrt(v)

    def _isCandidate(self, edge: Lattice.Edge) -> bool:
        """Edge parent→child is candidate if every parallel support edge is sound.

        For each predicate in parent, remove it and check that the same-predicate
        edge from that reduced parent is in soundEdges.
        """
        for pg in edge.fr().preds:
            if self._supportEdge(edge, pg) not in self.soundEdges:
                return False
        return True

    def propagateSound(self, edge: Lattice.Edge) -> None:
        """Given newly-sound edge A→X, find supersets of A (excluding X) and register
        any new candidate edges that add the same predicate from those supersets."""
        parent = edge.fr()
        lattice = self.lattice
        for pg2 in range(lattice.npgs):
            if pg2 in parent.preds or pg2 == edge.pg:
                continue
            for p2 in range(lattice.pgs[pg2]):
                superset = parent.to(pg2, p2).to()
                candidate_edge = superset.to(edge.pg, edge.p)
                if self._isCandidate(candidate_edge):
                    self.candidateEdges.add(candidate_edge)
