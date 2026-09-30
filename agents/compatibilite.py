MAX_ACCEPTABLE_DELAY = 20


def caregiver_is_eligible(caregiver: dict) -> bool:
    """Retourne True si le soignant peut recevoir une nouvelle affectation."""
    return (
        caregiver.get("available", False)
        and caregiver.get("delay", 0) <= MAX_ACCEPTABLE_DELAY
        and caregiver.get("current_workload", 0) < caregiver.get("max_work_hours", 0)
    )


def caregiver_matches_care_type(caregiver: dict, care_type: str) -> bool:
    """Accepte la compétence exacte ou la compétence polyvalente General."""
    return caregiver.get("skill") in (care_type, "General")


def caregiver_meets_skill_level(caregiver: dict, required_skill_level: int) -> bool:
    """Vérifie que le niveau de compétence du soignant couvre le besoin du patient."""
    return caregiver.get("skill_level", 1) >= required_skill_level


def caregiver_can_visit(
    caregiver: dict, care_type: str, required_skill_level: int = 1
) -> bool:
    """Centralise l'éligibilité, la compatibilité métier et le niveau de compétence."""
    return (
        caregiver_is_eligible(caregiver)
        and caregiver_matches_care_type(caregiver, care_type)
        and caregiver_meets_skill_level(caregiver, required_skill_level)
    )