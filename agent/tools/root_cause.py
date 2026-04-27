"""
Tool 8: RootCauseTool
Generate ranked root cause hypotheses based on WM-811K defect patterns and spatial analysis.
"""

import streamlit as st
from langchain_core.tools import tool

ROOT_CAUSE_DB = {
    "Center": {
        "hypotheses": [
            ("CMP non-uniformity", "Center material over-removal due to pad pressure distribution in Chemical Mechanical Planarisation."),
            ("Spin-coat centre effect", "Insufficient centrifugal force at wafer centre during photoresist spin-coat — thicker film residue."),
            ("Furnace temperature hotspot", "Localised high temperature at wafer centre during diffusion or anneal step."),
        ],
        "actions": [
            "Measure CMP uniformity map across multiple wafers — compare edge vs centre removal rate.",
            "Review spin-coat recipe: adjust RPM ramp-up profile to improve centre coverage.",
            "Run furnace temperature uniformity test; recalibrate heating zones.",
        ],
    },
    "Donut": {
        "hypotheses": [
            ("Spin-coat donut effect", "Solvent evaporation during spin-coat creates mid-radius photoresist ring."),
            ("Developer non-uniformity", "Developer pooling at mid-radius creates uneven develop rate."),
            ("Implant beam non-uniformity", "Annular pattern in ion dose distribution from beam steering anomaly."),
        ],
        "actions": [
            "Adjust spin coat exhaust airflow; review solvent vapour concentration above wafer.",
            "Change developer dispense nozzle pattern; increase dispense volume.",
            "Check ion implanter beam profile; recalibrate beam steering electromagnets.",
        ],
    },
    "Edge-Loc": {
        "hypotheses": [
            ("EBR malfunction", "Edge bead removal nozzle clogged or misaligned; photoresist accumulates at wafer edge."),
            ("Chuck clamping damage", "Mechanical stress at edge chuck contact points causes micro-fractures."),
            ("Wet process edge seal failure", "Chemical attack at wafer edge during wet etch or clean step."),
        ],
        "actions": [
            "Inspect EBR solvent nozzle; perform nozzle flush and realignment.",
            "Measure chuck contact force; replace worn chuck pins.",
            "Check wafer edge bevel protection during wet processing; adjust chemical concentration.",
        ],
    },
    "Edge-Ring": {
        "hypotheses": [
            ("Photolithography edge focus gradient", "Depth-of-focus limitation at wafer edge causes peripheral exposure failure."),
            ("RTP edge cooling", "Wafer edge cools faster than centre during rapid thermal processing — non-uniform anneal."),
            ("Plasma etch edge effect", "Gas flow non-uniformity at chamber wall causes over-etch at wafer edge ring."),
        ],
        "actions": [
            "Enable edge focus correction (EFC) in scanner; re-optimise focus offset at edge sites.",
            "Add edge ring (focus ring) to RTP chuck; recalibrate lamp power distribution.",
            "Adjust plasma etch gas flow and chamber pressure; optimise edge-to-centre uniformity.",
        ],
    },
    "Loc": {
        "hypotheses": [
            ("Particle contamination from process tool", "Localised particle source (robot, chuck, cassette) deposits on wafer surface."),
            ("Reticle defect", "Damaged photomask pattern repeats at same die location across multiple wafers."),
            ("Chuck particle transfer", "Debris on wafer chuck transfers to same XY position on each wafer."),
        ],
        "actions": [
            "Run particle map overlay across wafers — check if cluster location repeats (reticle) or random.",
            "Inspect reticle under SEM/optical microscope; compare defect coordinates to reticle pattern.",
            "Inspect and clean wafer chuck; check robot end effector for particle contamination.",
        ],
    },
    "Near-full": {
        "hypotheses": [
            ("Major process excursion", "Tool malfunction or process parameter out-of-control affecting entire wafer."),
            ("Chemical bath contamination", "Exhausted or cross-contaminated wet bath affecting all dies."),
            ("Photoresist coat failure", "Total coat failure — photoresist not dispensed correctly over wafer."),
        ],
        "actions": [
            "QUARANTINE lot immediately. Pull SPC charts for all process steps this lot passed through.",
            "Check chemical bath concentration, pH, and contamination levels; replace if exhausted.",
            "Review coat track log; check resist dispense nozzle and cup drain system.",
        ],
    },
    "Random": {
        "hypotheses": [
            ("Airborne particle contamination", "Elevated cleanroom particle count — HEPA filter integrity or human activity."),
            ("Water mark defects", "DI water droplets remaining after rinse-dry step leave residue on die surface."),
            ("Static charge discharge (ESD)", "Electrostatic discharge during wafer handling causes random die damage."),
        ],
        "actions": [
            "Check cleanroom particle counter — identify time correlation with process steps.",
            "Review wafer dry process: Marangoni dry, spin speed, and N2 blow-off conditions.",
            "Test ioniser output across all process tools; verify wrist-strap and ESD mat compliance.",
        ],
    },
    "Scratch": {
        "hypotheses": [
            ("Robot end effector contact", "SCARA/Bernoulli robot arm physically scratches wafer surface during transfer."),
            ("Cassette slot damage", "Worn or cracked cassette slot guide causes wafer to slide and scratch."),
            ("CMP pad conditioner fragment", "Diamond conditioner disc fragments embed in CMP pad — scratch subsequent wafers."),
        ],
        "actions": [
            "Inspect robot end effector for burrs or contamination; re-teach wafer transfer path.",
            "Visual inspection of all cassette slots; replace damaged FOUP/cassette.",
            "Inspect CMP pad for embedded conditioner fragments; replace pad and conditioner disc.",
        ],
    },
    "none": {
        "hypotheses": [
            ("Random parametric variation", "Fails are near electrical test limit — not a spatial/systematic issue."),
            ("Measurement noise", "Test equipment variation causing borderline dies to be incorrectly flagged."),
        ],
        "actions": [
            "Review electrical test limits — check if fails are just below spec limit (Cpk analysis).",
            "Run re-test on failed dies to determine reproducibility.",
        ],
    },
}


@tool
def root_cause_tool(defect_pattern: str = "auto") -> str:
    """
    Generate ranked root cause hypotheses for a given WM-811K defect pattern with recommended corrective actions.
    Input: defect_pattern — WM-811K label (Center/Donut/Edge-Loc/Edge-Ring/Loc/Near-full/Random/Scratch/none)
                            or 'auto' to automatically use the dominant defect from the loaded batch.
    Returns: ranked hypotheses with process-specific corrective actions.
    """
    try:
        if defect_pattern.strip().lower() == "auto":
            df = st.session_state.get("current_df")
            if df is None:
                return "ERROR: No batch loaded. Run data_ingestion_tool first."
            failed = df[df["pass_fail"] == 0]
            if len(failed) == 0:
                return "No failures in current batch. Yield is 100% — no root cause analysis needed."
            defect_pattern = failed["defect_code"].mode().iloc[0]

        entry = ROOT_CAUSE_DB.get(defect_pattern)

        if entry is None:
            return (
                f"Pattern '{defect_pattern}' not in WM-811K taxonomy.\n"
                "Valid patterns: Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch, none\n\n"
                "Generic recommendation:\n"
                "  1. Review SPC charts for all process steps this lot passed through.\n"
                "  2. Compare defect map with similar historical lots.\n"
                "  3. Escalate to process engineering for SEM/TEM cross-section analysis."
            )

        lines = [
            f"Root Cause Analysis — '{defect_pattern}' Pattern",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "",
            "Ranked Hypotheses:",
        ]

        for rank, (cause, explanation) in enumerate(entry["hypotheses"], 1):
            lines.append(f"  #{rank}. [{cause}]")
            lines.append(f"      {explanation}")

        lines += [
            "",
            "Recommended Corrective Actions:",
        ]
        for i, action in enumerate(entry["actions"], 1):
            lines.append(f"  {i}. {action}")

        lines += [
            "",
            "Note: Confirm hypothesis by cross-referencing with SPC data and process logs.",
        ]

        return "\n".join(lines)

    except Exception as e:
        return f"ERROR in root cause analysis: {type(e).__name__}: {str(e)}"