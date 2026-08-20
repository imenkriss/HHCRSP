from agents.agentsPatients import PatientAgent
from agents.agentsSoignants import soignantAgent


# ==========================================
# PATIENT
# ==========================================

patient = PatientAgent(

    patient_id="P1",

    priority="High",

    required_skill="Cardiology",

    preferred_caregiver="S2",

    location="Zone A",

    service_time=30,

    time_window=("08:00", "10:00")
)


# ==========================================
# CAREGIVER
# ==========================================

caregiver = soignantAgent(

    caregiver_id="S2",

    name="Caregiver S2",

    skills=[
        "Cardiology",
        "Diabetes"
    ],

    departure_time="08:00",

    workload=3,

    location="Zone A",

    availability=True,

    delay=5
)


# ==========================================
# PATIENT INFORMATION
# ==========================================

patient_information = patient.get_information()

print("\nPATIENT INFORMATION")
print(patient_information)


# ==========================================
# CAREGIVER INFORMATION
# ==========================================

print("\nCAREGIVER INFORMATION")
print(caregiver.get_information())


# ==========================================
# CAREGIVER EVALUATION
# ==========================================

print("\nCAREGIVER EVALUATION")

evaluation = caregiver.evaluate_patient(
    patient_information
)

print(evaluation)


# ==========================================
# PROPOSAL
# ==========================================

print("\nCARE PROPOSAL")

proposal = caregiver.generate_proposal(
    patient_information
)

print(proposal)


# ==========================================
# PATIENT RECEIVES PROPOSAL
# ==========================================

patient.receive_proposal(proposal)

print("\nPATIENT PROPOSALS")

patient.show_proposals()



import sys
import os

print("Python utilisé :", sys.executable)
print("Répertoire courant :", os.getcwd())
print("Chemin Python :")

for path in sys.path:
    print("  ", path)

import agents
print("PACKAGE AGENTS TROUVÉ :")
print(agents.__file__)