"""Prompt templates and phase label names for Cholec80-style classification."""

SYSTEM_PROMPT = (
    "You classify the current surgical phase from a single laparoscopic cholecystectomy video frame "
    "(Cholec80-style task). "
    "There are exactly seven phases, numbered 1-7: "
    "1 Preparation, 2 Calot triangle dissection, 3 Clipping and cutting, 4 Gallbladder dissection, "
    "5 Gallbladder packaging, 6 Cleaning and coagulation, 7 Gallbladder retraction. "
    "Reply with one digit only: 1, 2, 3, 4, 5, 6, or 7. No words, no punctuation, no explanation."
)

USER_PROMPT_TEACHER = "This phase belongs to "

PHASE_NAMES = {
    1: "Preparation",
    2: "CalotTriangleDissection",
    3: "ClippingCutting",
    4: "GallbladderDissection",
    5: "GallbladderPackaging",
    6: "CleaningCoagulation",
    7: "GallbladderRetraction",
}

label_dict = PHASE_NAMES
