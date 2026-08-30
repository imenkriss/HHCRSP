from complexity.metrics import ComplexityMetrics


metrics = ComplexityMetrics()


# ==========================================
# Test 1 : execution time
# ==========================================

start = metrics.start_timer()

# Simulation d'un traitement
total = 0

for i in range(100000):
    total += i

execution_time = metrics.stop_timer(start)

metrics.record_time(
    "test_component",
    execution_time
)


# ==========================================
# Test 2 : agent interactions
# ==========================================

interactions = metrics.calculate_agent_interactions(
    number_patients=16,
    number_caregivers=4
)


# ==========================================
# Results
# ==========================================

print("\n===== COMPLEXITY METRICS =====")

print(
    "Execution time:",
    execution_time,
    "seconds"
)

print(
    "Potential agent interactions:",
    interactions
)

print(
    "\nAll metrics:"
)

print(metrics.get_metrics())