"""
insights.py
-----------
Pure logic (no AI here) that turns raw detection counts into the
metrics and recommendations your app shows the user.
"""

def compute_insights(total_slots, occupied_slots):
    """
    Given how many slots were detected total and how many are occupied,
    return a dictionary with all the numbers and messages the app needs.
    """
    if total_slots == 0:
        return {
            "total_slots": 0,
            "occupied_slots": 0,
            "available_slots": 0,
            "occupancy_percent": 0.0,
            "congestion_level": "Unknown",
            "recommendation": "No parking slots were detected in this image."
        }

    available_slots = total_slots - occupied_slots
    occupancy_percent = round((occupied_slots / total_slots) * 100, 1)

    # Congestion thresholds (from your assignment brief)
    if occupancy_percent < 40:
        congestion_level = "Low"
        recommendation = "Plenty of slots available - proceed to park."
    elif occupancy_percent <= 75:
        congestion_level = "Moderate"
        recommendation = "Some slots available - proceed, but expect limited choice."
    else:
        congestion_level = "High"
        recommendation = "Parking nearly full - consider searching for another location."

    return {
        "total_slots": total_slots,
        "occupied_slots": occupied_slots,
        "available_slots": available_slots,
        "occupancy_percent": occupancy_percent,
        "congestion_level": congestion_level,
        "recommendation": recommendation
    }