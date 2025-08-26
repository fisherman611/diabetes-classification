import numpy as np

def years_to_years_months(age_years: float):
    """
    Convert age in float years into (years, months).
    Example: 5.5 -> (5, 6)
             10.25 -> (10, 3)
    """
    if age_years is None:
        return None
    
    years = int(age_years)  # whole years
    months = int(round((age_years - years) * 12))  # convert decimal to months
    
    return f"{years}-{months}"


def classify_bmi_row(row, bmi_percentile):
    """
    Classify BMI for one row depending on age:
    - Adults (>=20): use standard WHO cutoffs
    - Children (<20): use percentile table (p5, p85, p95)
    """
    bmi = row["bmi"]
    sex = row["gender"]
    age = row["age"]

    # Adult classification
    if age >= 20:
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal weight"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"

    # Child classification (lookup from bmi_percentile)
    key = row["age_scaled"].strip()
    match = bmi_percentile[
        (bmi_percentile["age_scaled"] == key) &
        (bmi_percentile["gender"] == sex)
    ]

    if match.empty:
        return np.nan  # no matching reference

    p5, p85, p95 = match[["p5","p85","p95"]].values[0]
    if bmi < p5:
        return "Underweight"
    elif bmi < p85:
        return "Normal weight"
    elif bmi < p95:
        return "Overweight"
    else:
        return "Obese"