import csv
import os
from typing import List, Dict, Tuple, Any

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def load_combined_dataset() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Charge exclusivement les fichiers existants dans le dossier data :
    - dementia_patients_health_data.csv pour les patients (avec ajout d'un 'id' et de la 'priority').
    - admin_caregivers.csv pour les soignants (avec ajout d'un 'id' et de la 'available').
    """
    patients = []
    soignants = []

    # 1. Chargement de votre fichier existant : dementia_patients_health_data.csv
    patient_path = os.path.join(DATA_DIR, "dementia_patients_health_data.csv")
    if os.path.exists(patient_path):
        with open(patient_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=1):
                # On conserve toutes vos données d'origine intactes sous forme de dictionnaire
                patient = {k.strip(): v.strip() for k, v in row.items() if k}
                
                # Ajout unique de l'ID patient demandé (ex: P1, P2...)
                patient["id"] = f"P{idx}"
                
                # Évaluation de la priorité par l'agent patient selon la tension, le diabète ou la démence
                diabetic = int(float(patient.get("Diabetic", 0)))
                dementia = int(float(patient.get("Dementia", 0)))
                condition = str(patient.get("Chronic_Health_Conditions", "")).lower()
                has_hypertension = "hypertension" in condition or "heart" in condition
                
                if diabetic == 1 or dementia == 1:
                    patient["priority"] = "High"
                elif has_hypertension:
                    patient["priority"] = "Medium"
                else:
                    patient["priority"] = "Less"

                patients.append(patient)

    # 2. Chargement de votre fichier existant : admin_caregivers.csv
    caregiver_path = os.path.join(DATA_DIR, "admin_caregivers.csv")
    if os.path.exists(caregiver_path):
        with open(caregiver_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=1):
                caregiver = {k.strip(): v.strip() for k, v in row.items() if k}
                
                # Ajout unique de l'ID soignant demandé (ex: S1, S2...)
                caregiver["id"] = f"S{idx}"
                
                # Gestion de la disponibilité du soignant
                caregiver["available"] = str(caregiver.get("available", "True")).lower() in ("true", "1", "yes", "oui")
                
                soignants.append(caregiver)

    return patients, soignants