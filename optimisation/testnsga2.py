from main import load_dataset
from optimisation.nsga2 import NSGA2


patients, soignants = load_dataset()

optimizer = NSGA2(
    patients,
    soignants,
    population_size=20,
    generations=10
)

solutions = optimizer.optimize()

print("\n===== NSGA-II =====")

print(
    "Solutions Pareto :",
    len(solutions)
)

for i, result in enumerate(solutions, 1):

    print(f"\nSolution {i}")

    print(
        "Objectifs :",
        result["objectives"]
    )

    for assignment in result["solution"]:

        print(
            f"  {assignment['patient_id']} "
            f"-> {assignment['caregiver_id']}"
        )