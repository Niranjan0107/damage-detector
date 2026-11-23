"""
Report Generator Module
Creates professional PDF and CSV Inspection Reports for Mine Operators.
"""
from typing import Dict, Any, List
import os
import pandas as pd
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


class ReportGenerator:
    @staticmethod
    def generate_csv(incidents: List[Dict[str, Any]], output_path: str) -> str:
        """Export incident records to a CSV file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        records = []
        for inc in incidents:
            records.append({
                "Vehicle ID": f"#{inc['vehicle_id']}",
                "Vehicle Class": inc["vehicle_class"],
                "First Detected (sec)": inc["first_detected_sec"],
                "Damage Types": ", ".join(inc["damage_types"]),
                "Max Severity (%)": inc["max_severity_score"],
                "Severity Level": inc["severity_level"],
                "Total Damage Points": inc["total_damage_points"],
                "Snapshot Saved": "Yes" if inc.get("snapshot_path") else "No",
            })
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        return output_path

    @staticmethod
    def generate_pdf(
        summary_stats: Dict[str, Any],
        incidents: List[Dict[str, Any]],
        output_pdf_path: str,
    ) -> str:
        """Export a comprehensive PDF inspection report with visual snapshot evidence."""
        os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)

        doc = SimpleDocTemplate(
            output_pdf_path,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Title"],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#1A365D"),
            alignment=0,
        )
        subtitle_style = ParagraphStyle(
            "Subtitle",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#4A5568"),
        )
        h2_style = ParagraphStyle(
            "Heading2",
            parent=styles["Heading2"],
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#2B6CB0"),
            spaceBefore=14,
            spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#2D3748"),
        )

        elements = []

        # 1. Header
        elements.append(Paragraph("MINING FLEET DAMAGE INSPECTION REPORT", title_style))
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elements.append(
            Paragraph(
                f"Generated on: <b>{now_str}</b> | System: <b>Mine CCTV Automated AI Sentry</b>",
                subtitle_style,
            )
        )
        elements.append(Spacer(1, 14))

        # 2. Executive Summary Metrics Table
        total_veh = summary_stats.get("total_vehicles_detected", 0)
        dmg_veh = summary_stats.get("damaged_vehicles_count", 0)
        critical_count = sum(1 for i in incidents if i["severity_level"] == "CRITICAL")
        major_count = sum(1 for i in incidents if i["severity_level"] == "MAJOR")

        summary_data = [
            ["Metric", "Value", "Status / Notes"],
            ["Total Vehicles Inspected", str(total_veh), "CCTV Stream Scanned"],
            ["Vehicles with Damage", str(dmg_veh), f"{round(dmg_veh / max(1, total_veh) * 100, 1)}% Fleet Defect Rate"],
            ["Critical Defects (Action Required)", str(critical_count), "Immediate Maintenance Recommended" if critical_count > 0 else "All Clear"],
            ["Major Defects", str(major_count), "Flagged for Secondary Inspection"],
        ]

        summary_table = Table(summary_data, colWidths=[200, 100, 240])
        summary_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F7FAFC")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ])
        )
        elements.append(summary_table)
        elements.append(Spacer(1, 16))

        # 3. Incident Breakdown Table
        elements.append(Paragraph("Detailed Defect Log", h2_style))

        if not incidents:
            elements.append(Paragraph("No vehicle body damage was detected in this footage. All units clear.", body_style))
        else:
            table_rows = [["Vehicle ID", "Class", "Time (s)", "Damage Detected", "Severity", "Action"]]
            for inc in incidents:
                sev_color = "#E53E3E" if inc["severity_level"] == "CRITICAL" else ("#DD6B20" if inc["severity_level"] == "MAJOR" else "#D69E2E")
                table_rows.append([
                    f"#{inc['vehicle_id']}",
                    inc["vehicle_class"],
                    f"{inc['first_detected_sec']}s",
                    "<br/>".join(inc["damage_types"]),
                    f"<font color='{sev_color}'><b>{inc['severity_level']} ({int(inc['max_severity_score'])}%)</b></font>",
                    "Ground Inspection" if inc["severity_level"] == "CRITICAL" else "Log & Monitor",
                ])

            # Convert row cells to Paragraphs for text wrapping
            formatted_rows = []
            for r_idx, row in enumerate(table_rows):
                formatted_row = []
                for cell in row:
                    if r_idx == 0:
                        formatted_row.append(Paragraph(f"<b>{cell}</b>", body_style))
                    else:
                        formatted_row.append(Paragraph(cell, body_style))
                formatted_rows.append(formatted_row)

            defect_table = Table(formatted_rows, colWidths=[70, 110, 50, 150, 80, 80])
            defect_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ])
            )
            elements.append(defect_table)

        elements.append(Spacer(1, 20))

        # 4. Photographic Evidence Gallery
        elements.append(Paragraph("Visual Photographic Evidence (Snapshots)", h2_style))
        elements.append(Paragraph("High-resolution cropped camera captures of flagged vehicles:", subtitle_style))
        elements.append(Spacer(1, 8))

        for inc in incidents[:6]:  # Show top 6 incidents in PDF
            snap_path = inc.get("snapshot_path")
            if snap_path and os.path.exists(snap_path):
                img_block = []
                label = f"<b>Vehicle #{inc['vehicle_id']} ({inc['vehicle_class']})</b> - Severity: {inc['severity_level']} ({int(inc['max_severity_score'])}%) | Damage: {', '.join(inc['damage_types'])}"
                img_block.append(Paragraph(label, body_style))
                img_block.append(Spacer(1, 4))
                try:
                    # Keep aspect ratio max width 400
                    rl_img = RLImage(snap_path, width=320, height=180)
                    img_block.append(rl_img)
                    img_block.append(Spacer(1, 10))
                    elements.append(KeepTogether(img_block))
                except Exception as e:
                    print(f"Failed to embed image {snap_path}: {e}")

        # Build document
        doc.build(elements)
        return output_pdf_path
