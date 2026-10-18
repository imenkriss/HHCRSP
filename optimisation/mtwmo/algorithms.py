"""Permutation-encoded multi-objective algorithms for the MTWMO-UnHHCRSP.

Implements the AGENTS.md requirements:

* P8  permutation chromosome (patients ``1..N``, caregiver separators
      ``N+1..N+K``), initial population 50% heuristic / 50% random;
* P9  NSGA-II (Deb et al. 2002) with fast non-dominated sort, crowding
      distance, binary tournament, constraint-domination, multimodal crossover
      (PMX + order + position based) at ``pc``, swap/insert mutation, archive
      cap, plus a deletion/insertion local search on elites;
* P10 the three objectives stay a vector (``(Z1, -Z2, Z3)``); the generation of
      the first feasible solution is recorded;
* MTCMOLite, a faithful-as-known re-implementation of the Qiao et al. (2022)
      two-cooperating-tasks idea (main constrained task + auxiliary
      epsilon-constraint task). It is NOT the original MTCMO; every deviation is
      documented in ``docs/ASSUMPTIONS.md``.

All randomness is drawn from a ``random.Random(seed)`` owned by ``run``.

ASSUMPTION (P8): the heuristic individuals pick, per patient, the highest-skill
compatible caregiver (ties broken by preference then randomly) and order each
route by earliest time-window start; patients with no compatible caregiver are
placed before the first separator (decoded as unassigned). Diversity among the
heuristic half comes from random tie-breaking.

ASSUMPTION (P9): mutation probability defaults to 0.1 (the paper table lists
only ``pc=0.9``); local search accepts a move when a constraint-first
lexicographic key improves, always keeping feasible solutions ahead of
infeasible ones.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .domain import Instance
from .model import Evaluation, evaluate

TOL = 1e-9


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #

@dataclass(eq=False)
class Candidate:
    """One permutation with its evaluated objectives and constraint status."""

    permutation: tuple[int, ...]
    objectives: tuple[float, float, float]
    violation: float = 0.0
    z1: float = 0.0
    z2: float = 0.0
    z3: float = 0.0
    feasible: bool = True
    generation: int = 0
    caregivers_used: int = 0
    evaluation: Evaluation | None = None

    @classmethod
    def from_evaluation(
        cls, permutation: Sequence[int], evaluation: Evaluation, generation: int = 0
    ) -> "Candidate":
        caregivers_used = sum(1 for detail in evaluation.routes if detail.patient_ids)
        return cls(
            permutation=tuple(int(gene) for gene in permutation),
            objectives=(evaluation.z1, -evaluation.z2, evaluation.z3),
            violation=evaluation.violations.total,
            z1=evaluation.z1,
            z2=evaluation.z2,
            z3=evaluation.z3,
            feasible=evaluation.violations.total <= TOL,
            generation=generation,
            caregivers_used=caregivers_used,
            evaluation=evaluation,
        )


@dataclass
class ParetoSet:
    """Result of one algorithm run (one seed, one instance, one gamma)."""

    algorithm: str
    instance_name: str
    seed: int
    gamma: object
    solutions: list[Candidate]
    first_feasible_generation: int | None
    generations: int
    runtime_seconds: float
    history: list[dict] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.solutions)

    @property
    def feasible_solutions(self) -> list[Candidate]:
        return [candidate for candidate in self.solutions if candidate.feasible]

    @property
    def has_feasible(self) -> bool:
        return any(candidate.feasible for candidate in self.solutions)

    @property
    def best_z1(self) -> float | None:
        return min((c.z1 for c in self.solutions), default=None)

    @property
    def best_z2(self) -> float | None:
        return max((c.z2 for c in self.solutions), default=None)

    @property
    def best_z3(self) -> float | None:
        return min((c.z3 for c in self.solutions), default=None)

    def _mean(self, attribute: str) -> float | None:
        if not self.solutions:
            return None
        return sum(getattr(c, attribute) for c in self.solutions) / len(self.solutions)

    @property
    def mean_z1(self) -> float | None:
        return self._mean("z1")

    @property
    def mean_z2(self) -> float | None:
        return self._mean("z2")

    @property
    def mean_z3(self) -> float | None:
        return self._mean("z3")

    @property
    def caregivers_used_mean(self) -> float | None:
        return self._mean("caregivers_used")


# --------------------------------------------------------------------------- #
# Dominance, ranking, crowding
# --------------------------------------------------------------------------- #

def pareto_dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """Classic Pareto dominance for minimisation vectors."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def constraint_dominates(
    a: Candidate,
    b: Candidate,
    feasibility: Callable[[Candidate], bool] | None = None,
) -> bool:
    """Constraint-domination (Deb 2000): feasible beats infeasible (P9).

    When both are infeasible the smaller total violation wins; ties fall back to
    Pareto dominance of the objectives.
    """
    is_feasible = feasibility if feasibility is not None else (lambda c: c.feasible)
    a_feasible, b_feasible = is_feasible(a), is_feasible(b)
    if a_feasible and not b_feasible:
        return True
    if b_feasible and not a_feasible:
        return False
    if not a_feasible and not b_feasible:
        if a.violation < b.violation - TOL:
            return True
        if b.violation < a.violation - TOL:
            return False
    return pareto_dominates(a.objectives, b.objectives)


def _fronts_indices(
    candidates: Sequence[Candidate], dominates: Callable[[Candidate, Candidate], bool]
) -> list[list[int]]:
    """Fast non-dominated sort (Deb et al. 2002), returning index fronts."""
    count = len(candidates)
    dominates_list: list[list[int]] = [[] for _ in range(count)]
    dominated_by = [0] * count
    fronts: list[list[int]] = [[]]
    for p in range(count):
        for q in range(count):
            if p == q:
                continue
            if dominates(candidates[p], candidates[q]):
                dominates_list[p].append(q)
            elif dominates(candidates[q], candidates[p]):
                dominated_by[p] += 1
        if dominated_by[p] == 0:
            fronts[0].append(p)
    current = 0
    while fronts[current]:
        nxt: list[int] = []
        for p in fronts[current]:
            for q in dominates_list[p]:
                dominated_by[q] -= 1
                if dominated_by[q] == 0:
                    nxt.append(q)
        current += 1
        fronts.append(nxt)
    fronts.pop()
    return fronts


def fast_non_dominated_sort(
    candidates: Sequence[Candidate],
    dominates: Callable[[Candidate, Candidate], bool] | None = None,
) -> list[list[Candidate]]:
    """Return Pareto fronts (list of Candidate lists) from best to worst."""
    rule = dominates if dominates is not None else constraint_dominates
    indices = _fronts_indices(candidates, rule)
    return [[candidates[index] for index in front] for front in indices]


def crowding_distance(front: Sequence[Candidate]) -> list[float]:
    """Deb's crowding distance; boundary points get +infinity."""
    size = len(front)
    if size <= 2:
        return [float("inf")] * size
    distance = [0.0] * size
    for objective in range(len(front[0].objectives)):
        order = sorted(range(size), key=lambda i: front[i].objectives[objective])
        distance[order[0]] = float("inf")
        distance[order[-1]] = float("inf")
        low = front[order[0]].objectives[objective]
        high = front[order[-1]].objectives[objective]
        if high - low <= TOL:
            continue
        for position in range(1, size - 1):
            previous = front[order[position - 1]].objectives[objective]
            following = front[order[position + 1]].objectives[objective]
            distance[order[position]] += (following - previous) / (high - low)
    return distance


def _rank_and_crowding(
    population: Sequence[Candidate],
    dominates: Callable[[Candidate, Candidate], bool] | None = None,
) -> tuple[list[int], list[float]]:
    """Per-individual rank and crowding distance for tournament selection."""
    fronts = _fronts_indices(population, dominates or constraint_dominates)
    rank = [0] * len(population)
    crowding = [0.0] * len(population)
    for level, front in enumerate(fronts):
        distances = crowding_distance([population[index] for index in front])
        for position, index in enumerate(front):
            rank[index] = level
            crowding[index] = distances[position]
    return rank, crowding


def binary_tournament(
    population: Sequence[Candidate], rank: Sequence[int], crowding: Sequence[float], rng: random.Random
) -> Candidate:
    """Binary tournament: lower rank, then higher crowding distance."""
    first = rng.randrange(len(population))
    second = rng.randrange(len(population))
    if rank[first] < rank[second]:
        return population[first]
    if rank[second] < rank[first]:
        return population[second]
    return population[first] if crowding[first] >= crowding[second] else population[second]


def environmental_selection(
    combined: Sequence[Candidate],
    size: int,
    dominates: Callable[[Candidate, Candidate], bool] | None = None,
) -> list[Candidate]:
    """Keep ``size`` candidates by front, then crowding distance (P9)."""
    fronts = fast_non_dominated_sort(combined, dominates)
    selected: list[Candidate] = []
    for front in fronts:
        if len(selected) + len(front) <= size:
            selected.extend(front)
            continue
        distances = crowding_distance(front)
        order = sorted(range(len(front)), key=lambda i: -distances[i])
        selected.extend(front[index] for index in order[: size - len(selected)])
        break
    return selected


# --------------------------------------------------------------------------- #
# Permutation operators (P8/P9)
# --------------------------------------------------------------------------- #

def random_permutation(patient_count: int, caregiver_count: int, rng: random.Random) -> list[int]:
    genes = list(range(1, patient_count + caregiver_count + 1))
    rng.shuffle(genes)
    return genes


def heuristic_permutation(instance: Instance, rng: random.Random) -> list[int]:
    """Heuristic individual: max-compatible skill, ordered by time windows."""
    patient_count = len(instance.patients)
    gene_of = {patient.id: index + 1 for index, patient in enumerate(instance.patients)}
    routes: dict[object, list] = {caregiver.id: [] for caregiver in instance.caregivers}
    leftovers: list[int] = []
    for patient in instance.patients:
        best_score: tuple[float, float] | None = None
        best_caregiver = None
        for caregiver in instance.caregivers:
            if caregiver.skill_level < patient.demand_level:
                continue
            if patient.rejections.get(caregiver.id) is True:
                continue
            score = (float(caregiver.skill_level), float(patient.preferences.get(caregiver.id, 0)))
            if best_score is None or score > best_score:
                best_score, best_caregiver = score, caregiver
            elif score == best_score and rng.random() < 0.5:
                best_caregiver = caregiver
        if best_caregiver is None:
            leftovers.append(gene_of[patient.id])
        else:
            routes[best_caregiver.id].append(patient)

    def window_key(patient) -> float:
        return min((window[0] for window in patient.time_windows), default=0.0)

    permutation = list(leftovers)
    for index, caregiver in enumerate(instance.caregivers):
        permutation.append(patient_count + 1 + index)
        for patient in sorted(routes[caregiver.id], key=window_key):
            permutation.append(gene_of[patient.id])
    return permutation


def pmx_crossover(parent_a: Sequence[int], parent_b: Sequence[int], rng: random.Random) -> list[int]:
    """Partially matched crossover (required by P9)."""
    size = len(parent_a)
    start, end = sorted(rng.sample(range(size), 2))
    child: list[int | None] = [None] * size
    child[start : end + 1] = list(parent_a[start : end + 1])
    mapping = {parent_a[i]: parent_b[i] for i in range(start, end + 1)}
    for index in list(range(0, start)) + list(range(end + 1, size)):
        gene = parent_b[index]
        while gene in child:
            gene = mapping[gene]
        child[index] = gene
    return [int(gene) for gene in child]


def order_crossover(parent_a: Sequence[int], parent_b: Sequence[int], rng: random.Random) -> list[int]:
    """Order crossover (OX1)."""
    size = len(parent_a)
    start, end = sorted(rng.sample(range(size), 2))
    segment = set(parent_a[start : end + 1])
    child: list[int | None] = [None] * size
    child[start : end + 1] = list(parent_a[start : end + 1])
    fill = [gene for gene in parent_b if gene not in segment]
    cursor = 0
    for index in range(size):
        if child[index] is None:
            child[index] = fill[cursor]
            cursor += 1
    return [int(gene) for gene in child]


def position_crossover(parent_a: Sequence[int], parent_b: Sequence[int], rng: random.Random) -> list[int]:
    """Position-based crossover (PBX)."""
    size = len(parent_a)
    chosen = set(rng.sample(range(size), rng.randint(1, size - 1)))
    kept = {parent_a[index] for index in chosen}
    child: list[int | None] = [None] * size
    for index in chosen:
        child[index] = parent_a[index]
    fill = [gene for gene in parent_b if gene not in kept]
    cursor = 0
    for index in range(size):
        if child[index] is None:
            child[index] = fill[cursor]
            cursor += 1
    return [int(gene) for gene in child]


CROSSOVERS: dict[str, Callable[[Sequence[int], Sequence[int], random.Random], list[int]]] = {
    "pmx": pmx_crossover,
    "ox": order_crossover,
    "pbx": position_crossover,
}


def swap_mutation(permutation: Sequence[int], rng: random.Random) -> list[int]:
    result = list(permutation)
    first, second = rng.sample(range(len(result)), 2)
    result[first], result[second] = result[second], result[first]
    return result


def insert_mutation(permutation: Sequence[int], rng: random.Random) -> list[int]:
    result = list(permutation)
    first, second = rng.sample(range(len(result)), 2)
    gene = result.pop(first)
    result.insert(second, gene)
    return result


def mutate(permutation: Sequence[int], rng: random.Random, probability: float) -> list[int]:
    if rng.random() >= probability:
        return list(permutation)
    return swap_mutation(permutation, rng) if rng.random() < 0.5 else insert_mutation(permutation, rng)


# --------------------------------------------------------------------------- #
# Local search (P9): deletion + insertion on elites
# --------------------------------------------------------------------------- #

def _constrained_key(candidate: Candidate) -> tuple:
    """Constraint-first lexicographic ordering used by the local search."""
    return (
        0 if candidate.feasible else 1,
        candidate.violation,
        candidate.z1,
        -candidate.z2,
        candidate.z3,
    )


def deletion_insertion_local_search(
    candidate: Candidate,
    instance: Instance,
    evaluate_permutation: Callable[[Sequence[int], int], Candidate],
    rng: random.Random,
    iterations: int,
) -> Candidate:
    """Improve one candidate by deleting a patient and reinserting it best.

    Every insertion position of the removed patient gene is tried; the best
    constraint-first move is accepted while it improves. Patient genes only are
    moved, so the chromosome stays a valid permutation.
    """
    patient_count = len(instance.patients)
    best = candidate
    for _ in range(iterations):
        topology = list(best.permutation)
        patient_genes = [gene for gene in topology if 1 <= gene <= patient_count]
        if not patient_genes:
            break
        gene = rng.choice(patient_genes)
        reduced = list(topology)
        reduced.remove(gene)
        best_move: Candidate | None = None
        for position in range(len(reduced) + 1):
            trial = reduced[:position] + [gene] + reduced[position:]
            evaluated = evaluate_permutation(trial, best.generation)
            if best_move is None or _constrained_key(evaluated) < _constrained_key(best_move):
                best_move = evaluated
        if best_move is not None and _constrained_key(best_move) < _constrained_key(best):
            best = best_move
        else:
            break
    return best


# --------------------------------------------------------------------------- #
# Common interface
# --------------------------------------------------------------------------- #

class Algorithm(ABC):
    """Common interface for the experiment runner (P10)."""

    name = "Algorithm"

    @abstractmethod
    def run(self, instance: Instance, seed: int = 0, gamma: object = 0.0) -> ParetoSet:
        raise NotImplementedError


def _update_archive(
    archive: Sequence[Candidate], incoming: Sequence[Candidate], archive_size: int
) -> list[Candidate]:
    """Non-dominated set of archive + incoming, capped by crowding distance."""
    seen: set[tuple[int, ...]] = set()
    merged: list[Candidate] = []
    for candidate in list(archive) + list(incoming):
        if candidate.permutation not in seen:
            seen.add(candidate.permutation)
            merged.append(candidate)
    fronts = fast_non_dominated_sort(merged)
    front = fronts[0] if fronts else []
    if len(front) > archive_size:
        distances = crowding_distance(front)
        order = sorted(range(len(front)), key=lambda i: -distances[i])
        front = [front[index] for index in order[:archive_size]]
    return front


class _PermutationAlgorithm(Algorithm):
    """Shared machinery: caching evaluator, init population, offspring, archive."""

    def __init__(
        self,
        population_size: int = 100,
        generations: int = 500,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        archive_size: int = 100,
        crossover_ops: Sequence[str] = ("pmx", "ox", "pbx"),
        seed: int | None = None,
    ) -> None:
        self.population_size = max(2, population_size)
        self.generations = max(0, generations)
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.archive_size = max(1, archive_size)
        unknown = set(crossover_ops) - set(CROSSOVERS)
        if unknown:
            raise ValueError(f"Unknown crossover operators: {sorted(unknown)}")
        self.crossover_ops = tuple(crossover_ops)
        self.seed = seed

    def _initial_population(self, instance: Instance, rng: random.Random) -> list[list[int]]:
        heuristic_count = self.population_size // 2
        population = [
            heuristic_permutation(instance, rng) for _ in range(heuristic_count)
        ]
        population.extend(
            random_permutation(len(instance.patients), len(instance.caregivers), rng)
            for _ in range(self.population_size - heuristic_count)
        )
        return population

    def _offspring(
        self,
        population: Sequence[Candidate],
        rank: Sequence[int],
        crowding: Sequence[float],
        rng: random.Random,
    ) -> list[list[int]]:
        children: list[list[int]] = []
        while len(children) < self.population_size:
            parent_a = binary_tournament(population, rank, crowding, rng)
            parent_b = binary_tournament(population, rank, crowding, rng)
            if rng.random() < self.crossover_prob:
                operator = CROSSOVERS[rng.choice(self.crossover_ops)]
                child_a = operator(parent_a.permutation, parent_b.permutation, rng)
                child_b = operator(parent_b.permutation, parent_a.permutation, rng)
            else:
                child_a = list(parent_a.permutation)
                child_b = list(parent_b.permutation)
            children.append(mutate(child_a, rng, self.mutation_prob))
            if len(children) < self.population_size:
                children.append(mutate(child_b, rng, self.mutation_prob))
        return children


class NSGAII(_PermutationAlgorithm):
    """NSGA-II (Deb et al. 2002) on the permutation encoding (P8/P9)."""

    name = "NSGA-II"

    def __init__(
        self,
        population_size: int = 100,
        generations: int = 500,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        archive_size: int = 100,
        crossover_ops: Sequence[str] = ("pmx", "ox", "pbx"),
        seed: int | None = None,
        local_search: bool = True,
        local_search_elites: int = 3,
        local_search_interval: int = 10,
        local_search_iterations: int = 5,
        costs=None,
    ) -> None:
        super().__init__(
            population_size, generations, crossover_prob, mutation_prob,
            archive_size, crossover_ops, seed,
        )
        self.local_search = local_search
        self.local_search_elites = max(0, local_search_elites)
        self.local_search_interval = max(1, local_search_interval)
        self.local_search_iterations = max(0, local_search_iterations)
        self.costs = costs

    def run(self, instance: Instance, seed: int = 0, gamma: object = 0.0) -> ParetoSet:
        start = time.perf_counter()
        rng = random.Random(seed)
        cache: dict[tuple[int, ...], Candidate] = {}

        def evaluate_permutation(permutation: Sequence[int], generation: int) -> Candidate:
            key = tuple(int(gene) for gene in permutation)
            candidate = cache.get(key)
            if candidate is None:
                evaluation = evaluate(list(key), instance, gamma, costs=self.costs)
                candidate = Candidate.from_evaluation(key, evaluation, generation)
                cache[key] = candidate
            return candidate

        population = [
            evaluate_permutation(permutation, 0)
            for permutation in self._initial_population(instance, rng)
        ]
        archive = _update_archive([], population, self.archive_size)
        first_feasible = 0 if any(c.feasible for c in population) else None
        history: list[dict] = []

        for generation in range(1, self.generations + 1):
            rank, crowding = _rank_and_crowding(population)
            offspring = [
                evaluate_permutation(permutation, generation)
                for permutation in self._offspring(population, rank, crowding, rng)
            ]
            combined = population + offspring
            population = environmental_selection(combined, self.population_size)
            if self.local_search and generation % self.local_search_interval == 0:
                population = self._local_search(population, instance, evaluate_permutation, rng)
            archive = _update_archive(archive, population, self.archive_size)
            if first_feasible is None and any(c.feasible for c in population):
                first_feasible = generation
            history.append({
                "generation": generation,
                "front_size": len(archive),
                "feasible": any(c.feasible for c in population),
                "best_z1": min((c.z1 for c in population), default=None),
            })

        return ParetoSet(
            algorithm=self.name,
            instance_name=getattr(instance, "name", ""),
            seed=seed,
            gamma=gamma,
            solutions=archive,
            first_feasible_generation=first_feasible,
            generations=self.generations,
            runtime_seconds=time.perf_counter() - start,
            history=history,
        )

    def _local_search(self, population, instance, evaluate_permutation, rng) -> list[Candidate]:
        rank, crowding = _rank_and_crowding(population)
        order = sorted(range(len(population)), key=lambda i: (rank[i], -crowding[i]))
        improved = list(population)
        for index in order[: self.local_search_elites]:
            candidate = deletion_insertion_local_search(
                population[index], instance, evaluate_permutation, rng,
                self.local_search_iterations,
            )
            improved[index] = candidate
        return environmental_selection(improved, self.population_size)


class MTCMOLite(_PermutationAlgorithm):
    """Two-cooperating-tasks CMO (Qiao et al. 2022 idea) -- NOT the original MTCMO.

    The main population optimises the true constrained problem
    (constraint-domination). The auxiliary population solves the same three
    objectives under an epsilon-constraint relaxation: any individual whose
    total violation is <= epsilon counts as feasible. Epsilon decays linearly
    from the initial maximum violation to zero, and the two populations exchange
    elites periodically. All deviations from the original are documented in
    ``docs/ASSUMPTIONS.md``.
    """

    name = "MTCMOLite"

    def __init__(
        self,
        population_size: int = 100,
        generations: int = 500,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        archive_size: int = 100,
        crossover_ops: Sequence[str] = ("pmx", "ox", "pbx"),
        seed: int | None = None,
        epsilon_fraction: float = 0.5,
        transfer_interval: int = 10,
        costs=None,
    ) -> None:
        super().__init__(
            population_size, generations, crossover_prob, mutation_prob,
            archive_size, crossover_ops, seed,
        )
        if not 0.0 < epsilon_fraction <= 1.0:
            raise ValueError("epsilon_fraction must be in (0, 1]")
        self.epsilon_fraction = epsilon_fraction
        self.transfer_interval = max(1, transfer_interval)
        self.costs = costs

    def run(self, instance: Instance, seed: int = 0, gamma: object = 0.0) -> ParetoSet:
        start = time.perf_counter()
        rng = random.Random(seed)
        cache: dict[tuple[int, ...], Candidate] = {}

        def evaluate_permutation(permutation: Sequence[int], generation: int) -> Candidate:
            key = tuple(int(gene) for gene in permutation)
            candidate = cache.get(key)
            if candidate is None:
                evaluation = evaluate(list(key), instance, gamma, costs=self.costs)
                candidate = Candidate.from_evaluation(key, evaluation, generation)
                cache[key] = candidate
            return candidate

        main = [
            evaluate_permutation(permutation, 0)
            for permutation in self._initial_population(instance, rng)
        ]
        auxiliary = [
            evaluate_permutation(permutation, 0)
            for permutation in self._initial_population(instance, rng)
        ]
        epsilon_initial = max((c.violation for c in main + auxiliary), default=0.0)
        epsilon_generations = max(1, int(self.epsilon_fraction * max(1, self.generations)))
        archive = _update_archive([], main, self.archive_size)
        first_feasible = 0 if any(c.feasible for c in main) else None
        history: list[dict] = []

        for generation in range(1, self.generations + 1):
            epsilon = epsilon_initial * max(0.0, (epsilon_generations - generation) / epsilon_generations)
            epsilon_feasible = (lambda c, e=epsilon: c.violation <= e + TOL)
            epsilon_dominates = (
                lambda a, b, f=epsilon_feasible: constraint_dominates(a, b, f)
            )

            main_rank, main_crowding = _rank_and_crowding(main)
            main_offspring = [
                evaluate_permutation(permutation, generation)
                for permutation in self._offspring(main, main_rank, main_crowding, rng)
            ]
            main = environmental_selection(main + main_offspring, self.population_size)

            aux_rank, aux_crowding = _rank_and_crowding(auxiliary, epsilon_dominates)
            aux_offspring = [
                evaluate_permutation(permutation, generation)
                for permutation in self._offspring(auxiliary, aux_rank, aux_crowding, rng)
            ]
            auxiliary = environmental_selection(
                auxiliary + aux_offspring, self.population_size, epsilon_dominates
            )

            if generation % self.transfer_interval == 0:
                main = self._transfer(auxiliary, main, self.population_size)
                auxiliary = self._transfer(main, auxiliary, self.population_size)

            archive = _update_archive(archive, main, self.archive_size)
            if first_feasible is None and any(c.feasible for c in main):
                first_feasible = generation
            history.append({
                "generation": generation,
                "epsilon": epsilon,
                "front_size": len(archive),
                "feasible": any(c.feasible for c in main),
            })

        return ParetoSet(
            algorithm=self.name,
            instance_name=getattr(instance, "name", ""),
            seed=seed,
            gamma=gamma,
            solutions=archive,
            first_feasible_generation=first_feasible,
            generations=self.generations,
            runtime_seconds=time.perf_counter() - start,
            history=history,
        )

    @staticmethod
    def _transfer(source: Sequence[Candidate], target: Sequence[Candidate], size: int) -> list[Candidate]:
        """Move the best source individuals into the target, replacing the worst."""
        donors = sorted(source, key=_constrained_key)[: max(1, size // 10)]
        pooled = list(target) + donors
        seen: set[tuple[int, ...]] = set()
        unique: list[Candidate] = []
        for candidate in pooled:
            if candidate.permutation not in seen:
                seen.add(candidate.permutation)
                unique.append(candidate)
        return environmental_selection(unique, size)


__all__ = [
    "Algorithm",
    "CROSSOVERS",
    "Candidate",
    "MTCMOLite",
    "NSGAII",
    "ParetoSet",
    "binary_tournament",
    "constraint_dominates",
    "crowding_distance",
    "deletion_insertion_local_search",
    "environmental_selection",
    "fast_non_dominated_sort",
    "heuristic_permutation",
    "insert_mutation",
    "order_crossover",
    "pareto_dominates",
    "pmx_crossover",
    "position_crossover",
    "random_permutation",
    "swap_mutation",
]
