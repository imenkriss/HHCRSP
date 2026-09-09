import random


class NSGA2:

    def __init__(
        self,
        patients,
        caregivers,
        population_size=20,
        generations=10
    ):
        self.patients = patients
        self.caregivers = caregivers
        self.population_size = population_size
        self.generations = generations

    # Vérifie si un soignant peut prendre le patient
    def is_compatible(self, patient, caregiver):

        if not caregiver.get("available", False):
            return False

        if caregiver.get("current_workload", 0) >= caregiver.get(
            "max_work_hours", 0
        ):
            return False

        return True

    # Crée une solution aléatoire
    def create_solution(self):

        solution = []

        for patient in self.patients:

            candidates = [
                caregiver
                for caregiver in self.caregivers
                if self.is_compatible(patient, caregiver)
            ]

            if candidates:

                caregiver = random.choice(candidates)

                solution.append({
                    "patient_id": patient["id"],
                    "caregiver_id": caregiver["id"]
                })

        return solution

    # Évalue une solution
    def evaluate(self, solution):

        total_delay = 0
        workload_difference = 0
        satisfaction = 0

        workloads = []

        for assignment in solution:

            caregiver = next(
                (
                    c for c in self.caregivers
                    if c["id"] == assignment["caregiver_id"]
                ),
                None
            )

            patient = next(
                (
                    p for p in self.patients
                    if p["id"] == assignment["patient_id"]
                ),
                None
            )

            if caregiver is None or patient is None:
                continue

            total_delay += caregiver.get("delay", 0)

            workloads.append(
                caregiver.get("current_workload", 0)
            )

            if (
                patient.get("preferred_caregiver")
                == caregiver.get("id")
            ):
                satisfaction += 1

        if workloads:
            workload_difference = (
                max(workloads) - min(workloads)
            )

        satisfaction_score = (
            satisfaction / len(self.patients)
            if self.patients
            else 0
        )

        # NSGA-II minimise tous les objectifs
        return [
            total_delay,
            workload_difference,
            -satisfaction_score
        ]

    # Vérifie si A domine B
    def dominates(self, a, b):

        return (
            all(x <= y for x, y in zip(a, b))
            and
            any(x < y for x, y in zip(a, b))
        )

    # Construit le front de Pareto
    def pareto_front(self, population):

        front = []

        for candidate in population:

            dominated = False

            for other in population:

                if candidate == other:
                    continue

                if self.dominates(
                    other["objectives"],
                    candidate["objectives"]
                ):
                    dominated = True
                    break

            if not dominated:
                front.append(candidate)

        return front

    # Exécute NSGA-II
    def optimize(self):

        population = []

        # Population initiale
        for _ in range(self.population_size):

            solution = self.create_solution()

            objectives = self.evaluate(solution)

            population.append({
                "solution": solution,
                "objectives": objectives
            })

        # Générations
        for _ in range(self.generations):

            new_population = []

            for _ in range(self.population_size):

                solution = self.create_solution()

                objectives = self.evaluate(solution)

                new_population.append({
                    "solution": solution,
                    "objectives": objectives
                })

            population.extend(new_population)

            population = self.pareto_front(
                population
            )

            population = population[
                :self.population_size
            ]

        return population

    def complexity_analysis(self):
        patients_count = len(self.patients)
        caregivers_count = len(self.caregivers)
        population_size = self.population_size
        generations = self.generations

        return {
            "patients": patients_count,
            "caregivers": caregivers_count,
            "population_size": population_size,
            "generations": generations,
            "evaluation_complexity": "O(N × C)",
            "pareto_front_complexity": "O(P²)",
            "total_complexity": "O(G × P × (N × C + P²))",
            "evaluation_operations": generations * population_size
            * patients_count * caregivers_count,
        }