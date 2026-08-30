RULES = [
    {
        "id": "R001",

        "category": "Priority",

        "keywords": [
            "urgent",
            "priority",
            "emergency"
        ],

        "condition":
        "patient.priority == HIGH",

        "action":
        "Schedule patient before all others.",

        "objective":
        "Patient Satisfaction",

        "description":
        "High-priority patients must always be served before lower-priority patients."
    },

    {
        "id": "R002",

        "category": "Caregiver Delay",

        "keywords": [
            "delay",
            "late",
            "retard"
        ],

        "condition":
        "caregiver.delay > 20",

        "action":
        "Reassign the patient to another available caregiver.",

        "objective":
        "Service Cost",

        "description":
        "A delayed caregiver can be replaced if the delay exceeds 20 minutes."
    }

]