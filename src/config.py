"""
Feature definitions for the Accident Severity Prediction project.
Separates features into Pre-Accident (Predictors) and Post-Accident (Outcome/Leakage).
"""

# Identifiers - to be dropped during training
IDENTIFIER_COLUMNS = [
    "Accident ID",
    "Region",
    "Zone",
    "Woreda (Sub-city)",
    "Location Easting",
    "Location Northing",
    "Date", # Temporal identifier, usually dropped in favor of Day/Time features
]

# Pre-Accident Features: Characteristics of the driver, vehicle, road, and environment.
# These are theoretically available at the moment of the crash (or immediately before).
PRE_ACCIDENT_FEATURES = [
    "Driver age",
    "Driver Sex",
    "Driver Education Level",
    "Driver experiance(years)",
    "Driver Vehicle Relation",
    "Driver's Action",
    "Vehicle Type",
    "Veh Ownership",
    "Veh Year of Service",
    "Vehicle Defects",
    "Land Use",
    "Road Type",
    "Road Surface",
    "Road Character",
    "Junction Type",
    "Light Condition",
    "Weather Condition",
    "Primary Collision type",
    "Time",
    "Day of the week",
    "Amet",  # Locale specific (likely Amharic for date/time aspect)
    "Wor",   # Locale specific
    "Number of involved vehicles", # Event characteristic
]

# Post-Accident Features: Outcome measures, injuries, damages.
# LEAKAGE - Do not use for real-time prediction. Only for upper-bound estimation.
POST_ACCIDENT_FEATURES = [
    "Number of fatalities",
    "Number of sever injuries",
    "Number of minor injuries",
    "Estimated Property damage",
    "Number of Casualties",
    "Victim-1 Type",
    "Victim-2 Type",
    "Victim-3 Type",
    "Victim-1 Sex",
    "Victim-2 Sex",
    "Victim-3 Sex",
    "Victim-1 age",
    "Victim-2 age",
    "Victim-3 age",
    "Victim-1 Injury Severity",
    "Victim-2 Injury Severity",
    "Victim-3 Injury Severity",
    "Victim-1 Movement",
    "Victim-2 Movement",
    "Victim-3 Movement",
]

# Initial Target Label
TARGET_COL = "Accident Type"
CLEAN_TARGET_COL = "Accident Type (Cleaned)"

# Config for Label Cleaning
KEEP_LABELS = {"minor": "Minor", "serious": "Serious", "fatal": "Fatal"}
