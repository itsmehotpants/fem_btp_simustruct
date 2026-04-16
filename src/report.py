"""
SimuStruct AI V3 — PDF Report Generator
=========================================
Generates professional engineering analysis reports using ReportLab.
"""

import io
import os
from datetime import datetime
from typing import Dict, Optional

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm, inch
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph,
        Spacer, Image, PageBreak, HRFlowable,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

from src.materials import MATERIALS


def generate_report(
    result: Dict,
    params: Dict,
    material_key: str = "structural_steel_a36",
    screenshot_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> bytes:
    """
    Generate a professional PDF engineering analysis report.

    Args:
        result: Simulation result dict (from run_ai_inference or run_fem)
        params: Input parameters dict (geometry, load, etc.)
        material_key: Material used in simulation
        screenshot_path: Optional path to heatmap screenshot image
        output_path: Optional path to save PDF directly

    Returns:
        PDF as bytes
    """
    if not HAS_REPORTLAB:
        raise ImportError(
            "ReportLab is required for PDF generation. "
            "Install: pip install reportlab"
        )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=20*mm, bottomMargin=20*mm,
        leftMargin=20*mm, rightMargin=20*mm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Title"],
        fontSize=22, spaceAfter=6*mm,
        textColor=colors.HexColor("#1a1d27"),
    )
    heading_style = ParagraphStyle(
        "CustomHeading", parent=styles["Heading2"],
        fontSize=14, spaceAfter=4*mm, spaceBefore=6*mm,
        textColor=colors.HexColor("#4A90D9"),
    )
    normal_style = ParagraphStyle(
        "CustomNormal", parent=styles["Normal"],
        fontSize=10, spaceAfter=2*mm,
    )
    small_style = ParagraphStyle(
        "Small", parent=styles["Normal"],
        fontSize=8, textColor=colors.grey,
    )

    story = []

    # ========== HEADER ==========
    story.append(Paragraph("SimuStruct AI — Simulation Report", title_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"SimuStruct AI V3.0",
        small_style
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#4A90D9")))
    story.append(Spacer(1, 6*mm))

    # ========== MATERIAL PROPERTIES ==========
    mat = MATERIALS.get(material_key, {})
    story.append(Paragraph("1. Material Properties", heading_style))

    mat_data = [
        ["Property", "Value", "Unit"],
        ["Material", mat.get("display_name", material_key), "-"],
        ["Category", mat.get("category", "-"), "-"],
        ["Young's Modulus (E)", f"{mat.get('E', 0)/1e9:.1f}", "GPa"],
        ["Poisson's Ratio (ν)", f"{mat.get('nu', 0):.3f}", "-"],
        ["Yield Strength (σ_y)", f"{mat.get('sigma_y', 0)/1e6:.0f}", "MPa"],
        ["Ultimate Tensile (UTS)", f"{mat.get('UTS', 0)/1e6:.0f}", "MPa"],
        ["Density (ρ)", f"{mat.get('rho', 0)}", "kg/m³"],
        ["CTE (α)", f"{mat.get('alpha', 0)*1e6:.1f}", "×10⁻⁶/°C"],
        ["Fracture Toughness (K_IC)", f"{mat.get('K_IC', 0)/1e6:.0f}", "MPa√m"],
    ]

    mat_table = Table(mat_data, colWidths=[160, 100, 60])
    mat_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4A90D9")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("ALIGN", (2, 0), (2, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(mat_table)
    story.append(Spacer(1, 4*mm))

    # ========== GEOMETRY & LOADING ==========
    story.append(Paragraph("2. Geometry & Loading Configuration", heading_style))

    geometry = params.get("geometry", {})
    load = params.get("load", {})

    geo_data = [
        ["Parameter", "Value"],
        ["Plate Width", f"{geometry.get('plate_width', '-')} m"],
        ["Plate Height", f"{geometry.get('plate_height', '-')} m"],
        ["Load Type", f"{load.get('type', '-')}"],
        ["Load Magnitude", f"{load.get('magnitude', 0)/1e3:.1f} kN/m²"],
    ]

    holes = geometry.get("holes", [])
    for i, h in enumerate(holes):
        geo_data.append([f"Hole {i+1} — Rx, Ry", f"{h.get('rx', '-')}, {h.get('ry', '-')} m"])
        geo_data.append([f"Hole {i+1} — Center (cx, cy)", f"({h.get('cx', '-')}, {h.get('cy', '-')})"])

    geo_table = Table(geo_data, colWidths=[180, 140])
    geo_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3DD68C")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0fff0")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(geo_table)
    story.append(Spacer(1, 4*mm))

    # ========== SIMULATION RESULTS ==========
    story.append(Paragraph("3. Simulation Results", heading_style))

    sf = result.get("safety_factor", 0)
    sf_status = "✓ SAFE" if sf > 1.5 else "⚠ WARNING" if sf > 1.0 else "✗ FAILURE"

    results_data = [
        ["Metric", "Value", "Status"],
        ["Max Von Mises Stress", f"{result.get('sigma_max_mpa', 0):.2f} MPa", "-"],
        ["Stress Concentration Factor (K_t)", f"{result.get('scf', 0):.3f}", "-"],
        ["Safety Factor (σ_y / σ_max)", f"{sf:.2f}", sf_status],
        ["AI Inference Time", f"{result.get('inference_ms', 0):.1f} ms", "-"],
        ["Number of Nodes", f"{result.get('n_nodes', 0):,}", "-"],
    ]

    if result.get("fem_time_ms"):
        speedup = result["fem_time_ms"] / max(result.get("inference_ms", 1), 0.1)
        results_data.append(["FEM Solver Time", f"{result['fem_time_ms']:.1f} ms", "-"])
        results_data.append(["AI Speedup", f"{speedup:.0f}×", "-"])

    if result.get("error_pct") is not None:
        results_data.append(["Mean Error vs FEM", f"{result['error_pct']:.2f}%", "-"])

    res_table = Table(results_data, colWidths=[180, 100, 80])
    res_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FF9F1C")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fff8f0")]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(res_table)

    # ========== SCREENSHOT ==========
    if screenshot_path and os.path.exists(screenshot_path):
        story.append(Spacer(1, 6*mm))
        story.append(Paragraph("4. Stress Field Visualization", heading_style))
        img = Image(screenshot_path, width=160*mm, height=100*mm)
        story.append(img)

    # ========== FOOTER ==========
    story.append(Spacer(1, 10*mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Paragraph(
        "This report was generated by SimuStruct AI V3.0 — "
        "Deep Learning Structural Analysis Platform. "
        "Results are approximate and should be validated with full FEM analysis "
        "for safety-critical applications.",
        small_style
    ))
    story.append(Paragraph(
        "© 2024-2027 LNMIIT Jaipur — B.Tech Project",
        small_style,
    ))

    # Build PDF
    doc.build(story)
    pdf_bytes = buf.getvalue()

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    return pdf_bytes
