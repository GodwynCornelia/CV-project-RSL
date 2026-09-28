import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.models import CandidateLattice, Hypothesis


@dataclass
class LatticeNode:
    """Узел графа решетки: конкретная жестовая гипотеза на шаге t."""
    step: int
    gloss: str
    score: float
    log_prob: float

    @property
    def id(self) -> str:
        return f"{self.step}_{self.gloss}"


@dataclass
class LatticeEdge:
    """Направленное взвешенное ребро между гипотезами шага t и t+1."""
    source_id: str
    target_id: str
    weight: float                    # -log P(target) + штраф перехода


@dataclass
class BeamPath:
    """Траектория пути в решетке гипотез."""
    nodes: List[LatticeNode] = field(default_factory=list)
    cumulative_cost: float = 0.0     # Сумма отрицательных логарифмов правдоподобия

    @property
    def glosses(self) -> List[str]:
        return [node.gloss for node in self.nodes]

    @property
    def total_score(self) -> float:
        # Экспонента от среднего логарифма
        if not self.nodes:
            return 0.0
        return math.exp(-self.cumulative_cost / max(1, len(self.nodes)))


class LatticeGraph:
    """
    Взвешенный ориентированный ациклический граф (Candidate Lattice DAG).
    Патентоспособный узел: формирование многомерной решетки распределений вероятностей
    и отбор путей максимального правдоподобия через Beam Search до вызова LLM.
    """

    def __init__(self, lattices: List[CandidateLattice]):
        self.lattices = lattices
        self.nodes_by_step: Dict[int, List[LatticeNode]] = {}
        self._build_graph()

    def _build_graph(self) -> None:
        for step, lat in enumerate(self.lattices):
            self.nodes_by_step[step] = []
            for hyp in lat.candidates:
                # Численная стабильность для log
                clipped_score = max(hyp.score, 1e-6)
                log_p = -math.log(clipped_score)
                node = LatticeNode(
                    step=step,
                    gloss=hyp.gloss,
                    score=hyp.score,
                    log_prob=log_p
                )
                self.nodes_by_step[step].append(node)

    def beam_search(self, beam_width: int = 3) -> List[BeamPath]:
        """
        Поиск путей максимального правдоподобия через решетку гипотез.
        Снижает нагрузку на языковую модель, отсеивая шумовые комбинации.
        """
        if not self.nodes_by_step:
            return []

        # Инициализация путей первого шага (step = 0)
        first_step_nodes = self.nodes_by_step.get(0, [])
        if not first_step_nodes:
            return []

        beams: List[BeamPath] = [
            BeamPath(nodes=[node], cumulative_cost=node.log_prob)
            for node in first_step_nodes
        ]
        beams.sort(key=lambda p: p.cumulative_cost)
        beams = beams[:beam_width]

        # Последовательное расширение путей по шагам
        num_steps = len(self.nodes_by_step)
        for step in range(1, num_steps):
            next_nodes = self.nodes_by_step.get(step, [])
            candidate_paths: List[BeamPath] = []

            for path in beams:
                last_node = path.nodes[-1]
                for next_node in next_nodes:
                    # Штраф за мгновенный повтор одного и того же жеста (анти-статтеринг)
                    repetition_penalty = 1.2 if next_node.gloss == last_node.gloss else 0.0

                    new_cost = path.cumulative_cost + next_node.log_prob + repetition_penalty
                    new_path = BeamPath(
                        nodes=path.nodes + [next_node],
                        cumulative_cost=new_cost
                    )
                    candidate_paths.append(new_path)

            candidate_paths.sort(key=lambda p: p.cumulative_cost)
            beams = candidate_paths[:beam_width]

        return beams

    def to_compact_representation(self) -> List[dict]:
        """Преобразует решетку в компактный формат для системного промпта LLM."""
        return [lat.to_dict() for lat in self.lattices]
