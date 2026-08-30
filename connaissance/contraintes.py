"""
Operational constraints of the HHCOP.
"""

CONSTRAINTS = [

    {
        "id": "C001",
        "category": "Visit",
        "keywords": [
            "visit",
            "patient",
            "once"
        ],
        "description": "Each patient must be visited exactly once."
    },

    {
        "id": "C002",
        "category": "Workload",
        "keywords": [
            "hours",
            "workload",
            "overtime",
            "charge"
        ],
        "description": "A caregiver cannot exceed the maximum working hours."
    },

    {
        "id": "C003",
        "category": "Time Window",
        "keywords": [
            "time window",
            "schedule",
            "delay",
            "late",
            "retard"
        ],
        "description": "Patient time windows must be respected."
    },

    {
        "id": "C004",
        "category": "Qualification",
        "keywords": [
            "skill",
            "qualified",
            "specialized",
            "cardio",
            "diabetes"
        ],
        "description": "Only qualified caregivers can perform specialized care."
    }

]
