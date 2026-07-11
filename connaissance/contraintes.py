"""
Operational constraints of the HHCOP.
"""

CONSTRAINTS = [

    {
        "id": "C001",
        "description": "Each patient must be visited exactly once."
    },

    {
        "id": "C002",
        "description": "A caregiver cannot exceed the maximum working hours."
    },

    {
        "id": "C003",
        "description": "Patient time windows must be respected."
    },

    {
        "id": "C004",
        "description": "Only qualified caregivers can perform specialized care."
    }

]