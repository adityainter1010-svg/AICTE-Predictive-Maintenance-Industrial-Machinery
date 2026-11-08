"""
src/recommendations.py
Generate maintenance recommendations based on predicted failure type and sensor values.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ── Recommendation Lookup Table ───────────────────────────────────────────────

_RECOMMENDATIONS = {
    "No Failure": {
        "urgency": "Routine",
        "actions": [
            "Continue standard preventive maintenance schedule.",
            "Log current sensor readings for trend analysis.",
            "Verify lubrication levels are within specification.",
            "Inspect machine guards and safety interlocks.",
        ],
        "timeline": "Next scheduled maintenance window",
    },
    "Tool Wear Failure": {
        "urgency": "Immediate",
        "actions": [
            "🔴 Stop machine and replace cutting tool immediately.",
            "Inspect tool holder and spindle for damage.",
            "Review tool-change intervals; reduce by 15-20%.",
            "Check surface-finish quality of recent production runs.",
            "Log tool wear patterns for predictive scheduling.",
        ],
        "timeline": "Within the next 1-2 hours",
    },
    "Heat Dissipation Failure": {
        "urgency": "Urgent",
        "actions": [
            "🟠 Reduce process temperature by adjusting coolant flow rate.",
            "Clean heat exchangers and cooling fins.",
            "Verify that air temperature in machine bay is within limits.",
            "Inspect temperature sensors for calibration drift.",
            "Ensure the process temperature-air temperature difference stays below 10 K.",
        ],
        "timeline": "Within the next 4-8 hours",
    },
    "Power Failure": {
        "urgency": "Immediate",
        "actions": [
            "🔴 Halt machine and inspect power supply unit.",
            "Check motor winding insulation resistance.",
            "Measure supply voltage and current under load; compare to rated values.",
            "Inspect power cables, connectors, and circuit breakers.",
            "Verify torque x rpm product is within rated power envelope.",
        ],
        "timeline": "Before next production cycle",
    },
    "Overstrain Failure": {
        "urgency": "Urgent",
        "actions": [
            "🟠 Reduce applied torque or cutting depth immediately.",
            "Inspect drive system (gears, belts, couplings) for wear.",
            "Check machine alignment and fixture clamping.",
            "Review process parameters - consider lighter machining passes.",
            "Evaluate bearing and spindle load capacity.",
        ],
        "timeline": "Within the next 2-4 hours",
    },
    "Random Failures": {
        "urgency": "High",
        "actions": [
            "Schedule comprehensive diagnostic inspection.",
            "Review vibration and acoustic emission data.",
            "Check for loose fasteners, worn bearings, and seal integrity.",
            "Inspect lubrication system for contamination or low levels.",
            "Consider thermal imaging scan of critical components.",
        ],
        "timeline": "Within the next 24 hours",
    },
    "Multiple Failures": {
        "urgency": "Critical – Immediate Shutdown",
        "actions": [
            "🔴 STOP machine immediately – multiple failure modes detected.",
            "Do not restart until full diagnostic is completed.",
            "Notify maintenance supervisor and safety officer.",
            "Isolate power, pneumatic, and hydraulic supplies.",
            "Perform root-cause analysis before returning to service.",
        ],
        "timeline": "Immediate – do not delay",
    },
}

# Short-code aliases pointing to the same recommendations
_RECOMMENDATIONS["TWF"] = _RECOMMENDATIONS["Tool Wear Failure"]
_RECOMMENDATIONS["HDF"] = _RECOMMENDATIONS["Heat Dissipation Failure"]
_RECOMMENDATIONS["PWF"] = _RECOMMENDATIONS["Power Failure"]
_RECOMMENDATIONS["OSF"] = _RECOMMENDATIONS["Overstrain Failure"]
_RECOMMENDATIONS["RNF"] = _RECOMMENDATIONS["Random Failures"]
_RECOMMENDATIONS["Multiple"] = _RECOMMENDATIONS["Multiple Failures"]

URGENCY_COLORS = {
    "Routine": "#2ca02c",
    "Urgent": "#ff7f0e",
    "Immediate": "#d62728",
    "High": "#ff7f0e",
    "Critical – Immediate Shutdown": "#7f0000",
}


def get_recommendations(failure_type_code: str, sensor_values: dict = None) -> dict:
    """
    Return a recommendation dict for the given failure type code.
    Optionally enriches with sensor-specific advice.
    """
    rec = _RECOMMENDATIONS.get(failure_type_code, _RECOMMENDATIONS["RNF"]).copy()
    rec["failure_type_code"] = failure_type_code
    rec["failure_type_label"] = config.FAILURE_TYPE_MAP.get(failure_type_code, failure_type_code)
    rec["urgency_color"] = URGENCY_COLORS.get(rec["urgency"], "#7f7f7f")

    # Sensor-specific additional advice
    if sensor_values:
        extra = _sensor_specific_advice(sensor_values)
        if extra:
            rec["actions"] = rec["actions"] + extra

    return rec


def _sensor_specific_advice(sensor_values: dict) -> list:
    """Generate extra advice lines based on out-of-range sensor values."""
    advice = []

    tool_wear = sensor_values.get("Tool wear [min]") or sensor_values.get("Tool Wear (min)")
    if tool_wear is not None:
        if tool_wear >= config.TOOL_WEAR_CRITICAL_THRESHOLD:
            advice.append(f"⚠️ Tool wear ({tool_wear} min) is CRITICAL – replace immediately.")
        elif tool_wear >= config.TOOL_WEAR_WARNING_THRESHOLD:
            advice.append(f"⚠️ Tool wear ({tool_wear} min) approaching limit – plan replacement soon.")

    torque = sensor_values.get("Torque [Nm]") or sensor_values.get("Torque (Nm)")
    if torque is not None and torque > config.TORQUE_WARNING_THRESHOLD:
        advice.append(f"⚠️ Torque ({torque} Nm) exceeds warning threshold ({config.TORQUE_WARNING_THRESHOLD} Nm).")

    air_temp = sensor_values.get("Air temperature [K]") or sensor_values.get("Air Temp (K)")
    proc_temp = sensor_values.get("Process temperature [K]") or sensor_values.get("Process Temp (K)")
    if air_temp and proc_temp:
        diff = proc_temp - air_temp
        if diff > config.TEMP_DIFF_WARNING_THRESHOLD:
            advice.append(
                f"⚠️ Temperature differential ({diff:.1f} K) exceeds safe limit "
                f"({config.TEMP_DIFF_WARNING_THRESHOLD} K). Check cooling system."
            )

    return advice


def format_recommendations_markdown(rec: dict) -> str:
    """Format recommendations as a markdown string for Streamlit display."""
    lines = [
        f"### Maintenance Recommendations",
        f"**Failure Type:** {rec['failure_type_label']}",
        f"**Urgency:** {rec['urgency']}",
        f"**Recommended Timeline:** {rec['timeline']}",
        "",
        "**Actions Required:**",
    ]
    for i, action in enumerate(rec["actions"], 1):
        lines.append(f"{i}. {action}")
    return "\n".join(lines)
