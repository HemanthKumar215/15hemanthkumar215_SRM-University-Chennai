"""
ui_theme.py – Health-Tech Design System, Themes & Vector Graphics
==================================================================
Palette:
  - Midnight navy: #0B1220
  - Surface: #121B2E
  - Primary Teal: #2DD4BF
  - Cyan: #38BDF8
  - Accent Violet: #8B5CF6
  - In Range Green: #34D399
  - Watch Amber: #FBBF24
  - High Coral: #FB7185
  - Text Light: #F8FAFC, Slate: #94A3B8
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Color Tokens
# ══════════════════════════════════════════════════════════════════════════════

THEME_COLORS = {
    "dark": {
        "bg": "#0B1220",
        "surface": "#121B2E",
        "surface_card": "rgba(18, 27, 46, 0.75)",
        "surface_card_border": "rgba(56, 189, 248, 0.14)",
        "surface_hover": "rgba(30, 41, 68, 0.85)",
        "text_primary": "#F8FAFC",
        "text_muted": "#94A3B8",
        "text_subtle": "#64748B",
        "primary": "#2DD4BF",
        "cyan": "#38BDF8",
        "violet": "#8B5CF6",
        "green": "#34D399",
        "amber": "#FBBF24",
        "coral": "#FB7185",
        "grid": "rgba(148, 163, 184, 0.08)",
        "plot_bg": "rgba(15, 23, 42, 0.35)",
    },
    "light": {
        "bg": "#F6F8FB",
        "surface": "#FFFFFF",
        "surface_card": "#FFFFFF",
        "surface_card_border": "#E2E8F0",
        "surface_hover": "#F1F5F9",
        "text_primary": "#0F172A",
        "text_muted": "#475569",
        "text_subtle": "#64748B",
        "primary": "#0D9488",
        "cyan": "#0284C7",
        "violet": "#7C3AED",
        "green": "#059669",
        "amber": "#D97706",
        "coral": "#E11D48",
        "grid": "rgba(15, 23, 42, 0.08)",
        "plot_bg": "#FFFFFF",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Vector Icons (Clean Lucide / Material Style SVGs)
# ══════════════════════════════════════════════════════════════════════════════

def svg_icon(name: str, color: str = "#2DD4BF", size: int = 18) -> str:
    """Return crisp inline SVG icon markup with consistent styling."""
    icons = {
        "dna": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m8 16 1.5-1.5"/><path d="m14.5 9.5 1.5-1.5"/><path d="m14 16 1.5-1.5"/><path d="m8.5 9.5 1.5-1.5"/><path d="m14 8 1.5-1.5"/><path d="m8.5 15.5 1.5-1.5"/><path d="M2 15c6.667-6 13.333 0 20-6"/><path d="M9 22c1.798-1.998 2.518-3.995 2.807-5.993"/><path d="M15 2c-1.798 1.998-2.518 3.995-2.807 5.993"/><path d="m17 6-2.5-2.5"/><path d="m7 18 2.5 2.5"/></svg>''',
        "drop": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/></svg>''',
        "heart": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/></svg>''',
        "pulse": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>''',
        "activity": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 14l6-6 4 4 6-6"/><path d="M14 6h6v6"/></svg>''',
        "walk": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M13 4a2 2 0 1 0-4 0 2 2 0 0 0 4 0Z"/><path d="m7 21 3-9 2.5 3.5L16 21"/><path d="m9 12 3-3 3 2.5 3-1.5"/></svg>''',
        "moon": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg>''',
        "sun": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/></svg>''',
        "gauge": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/></svg>''',
        "shield": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10"/><path d="m9 12 2 2 4-4"/></svg>''',
        "sparkles": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"/></svg>''',
        "alert": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>''',
        "check": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>''',
        "trending_up": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>''',
        "trending_down": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/></svg>''',
        "trending_flat": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="2" y1="12" x2="22" y2="12"/><polyline points="16 6 22 12 16 18"/></svg>''',
        "pill": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m10.5 20.5 10-10a4.95 4.95 0 1 0-7-7l-10 10a4.95 4.95 0 1 0 7 7Z"/><path d="m8.5 8.5 7 7"/></svg>''',
        "hospital": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 6v4"/><path d="M14 8h-4"/><path d="M18 18V4a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v14"/><path d="M2 22h20"/><path d="M10 22v-4a2 2 0 0 1 4 0v4"/></svg>''',
        "upload": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>''',
        "download": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>''',
        "sliders": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>''',
    }
    return icons.get(name, "")


# ══════════════════════════════════════════════════════════════════════════════
# SVG Sparkline Generator
# ══════════════════════════════════════════════════════════════════════════════

def generate_svg_sparkline(values: List[float], stroke_color: str = "#2DD4BF", width: int = 100, height: int = 28) -> str:
    """Generate a lightweight, smooth inline SVG sparkline."""
    if not values or len(values) < 2:
        return ""
    vals = np.array(values)
    v_min, v_max = float(vals.min()), float(vals.max())
    span = max(1.0, v_max - v_min)
    n = len(vals)
    pts = []
    for i, v in enumerate(vals):
        x = (i / (n - 1)) * (width - 6) + 3
        y = height - 4 - ((v - v_min) / span) * (height - 8)
        pts.append(f"{x:.1f},{y:.1f}")
    polyline_pts = " ".join(pts)
    return f'<svg width="{width}" height="{height}" style="overflow:visible;vertical-align:middle;"><polyline fill="none" stroke="{stroke_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" points="{polyline_pts}"/></svg>'


def generate_svg_semicircle_gauge(prob_pct: float, risk_level: str = "low", width: int = 120, height: int = 60) -> str:
    """Generate an inline SVG semicircle animated gauge with green/amber/coral zones."""
    pct = max(0.0, min(100.0, float(prob_pct)))
    dash_total = 131.95  # pi * r (r=42)
    dash_offset = dash_total * (1.0 - (pct / 100.0))
    needle_color = "#34D399" if pct <= 25 else ("#FBBF24" if pct <= 50 else "#FB7185")

    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 110 56" style="overflow:visible;vertical-align:middle;">'
        f'<defs>'
        f'<linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">'
        f'<stop offset="0%" stop-color="#34D399"/>'
        f'<stop offset="35%" stop-color="#FBBF24"/>'
        f'<stop offset="85%" stop-color="#FB7185"/>'
        f'</linearGradient>'
        f'</defs>'
        f'<path d="M 13 48 A 42 42 0 0 1 97 48" fill="none" stroke="rgba(148, 163, 184, 0.15)" stroke-width="8" stroke-linecap="round"/>'
        f'<path d="M 13 48 A 42 42 0 0 1 97 48" fill="none" stroke="url(#gaugeGrad)" stroke-width="8" stroke-linecap="round" opacity="0.30"/>'
        f'<path d="M 13 48 A 42 42 0 0 1 97 48" fill="none" stroke="{needle_color}" stroke-width="8" stroke-linecap="round" stroke-dasharray="{dash_total}" stroke-dashoffset="{dash_offset}"/>'
        f'<text x="55" y="44" text-anchor="middle" fill="{needle_color}" font-family="Plus Jakarta Sans, sans-serif" font-size="16" font-weight="800">{pct:.0f}%</text>'
        f'<text x="55" y="55" text-anchor="middle" fill="#94A3B8" font-family="Inter, sans-serif" font-size="8" font-weight="600">{risk_level.upper()}</text>'
        f'</svg>'
    )



# ══════════════════════════════════════════════════════════════════════════════
# CSS Stylesheet Generator
# ══════════════════════════════════════════════════════════════════════════════

def get_custom_css(theme_mode: str = "dark") -> str:
    c = THEME_COLORS.get(theme_mode, THEME_COLORS["dark"])
    is_dark = (theme_mode == "dark")

    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

    :root {{
        --bg: {c["bg"]};
        --surface: {c["surface"]};
        --surface-card: {c["surface_card"]};
        --surface-border: {c["surface_card_border"]};
        --text: {c["text_primary"]};
        --muted: {c["text_muted"]};
        --border: {c["surface_card_border"]};
        --accent: {c["primary"]};
        --cyan: {c["cyan"]};
        --violet: {c["violet"]};
        --green: {c["green"]};
        --amber: {c["amber"]};
        --coral: {c["coral"]};
    }}

    /* Global Typography & Hide Default Streamlit Chrome */
    #MainMenu {{ visibility: hidden !important; }}
    footer {{ visibility: hidden !important; }}
    header {{ visibility: hidden !important; }}
    .stDeployButton {{ display: none !important; }}

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, sans-serif !important;
        color: var(--text) !important;
    }}
    h1, h2, h3, h4, h5, h6, .display-font {{
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        letter-spacing: -0.02em;
        color: var(--text) !important;
    }}

    /* Full Page & Ambient Background */
    .stApp {{
        background: var(--bg) !important;
        background-image: {'radial-gradient(ellipse at 15% 10%, rgba(45, 212, 191, 0.08) 0%, transparent 45%), radial-gradient(ellipse at 85% 25%, rgba(139, 92, 246, 0.07) 0%, transparent 45%)' if is_dark else 'radial-gradient(ellipse at 15% 10%, rgba(13, 148, 136, 0.05) 0%, transparent 45%), radial-gradient(ellipse at 85% 25%, rgba(124, 58, 237, 0.04) 0%, transparent 45%)'};
        background-attachment: fixed;
    }}

    /* Sidebar Theme Enforcement */
    [data-testid="stSidebar"] {{
        background-color: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }}
    [data-testid="stSidebar"] * {{
        color: var(--text) !important;
    }}
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {{
        color: var(--text) !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="select"] div {{
        background-color: {'#1E293B' if is_dark else '#F1F5F9'} !important;
        color: var(--text) !important;
    }}

    /* Hero Section Gradient Container */
    .hero-container {{
        background: {'linear-gradient(135deg, rgba(18, 27, 46, 0.90) 0%, rgba(15, 23, 42, 0.95) 100%)' if is_dark else 'linear-gradient(135deg, #FFFFFF 0%, #F1F5F9 100%)'};
        border: 1px solid {c["surface_card_border"]};
        border-radius: 18px;
        padding: 20px 26px;
        margin-bottom: 20px;
        box-shadow: {'0 10px 30px rgba(0, 0, 0, 0.40)' if is_dark else '0 4px 20px rgba(15, 23, 42, 0.06)'};
        backdrop-filter: blur(14px);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 20px;
    }}

    .avatar-circle {{
        width: 58px;
        height: 58px;
        border-radius: 50%;
        background: linear-gradient(135deg, {c["primary"]} 0%, {c["violet"]} 100%);
        color: #0B1220;
        font-weight: 800;
        font-size: 1.35rem;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 20px {c["primary"]}44;
        flex-shrink: 0;
    }}

    /* Pulsing Status Pills */
    .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }}
    .pill-stable {{
        background: rgba(52, 211, 153, 0.16);
        color: {c["green"]};
        border: 1px solid rgba(52, 211, 153, 0.35);
        box-shadow: 0 0 12px rgba(52, 211, 153, 0.20);
    }}
    .pill-watch {{
        background: rgba(251, 191, 36, 0.16);
        color: {c["amber"]};
        border: 1px solid rgba(251, 191, 36, 0.35);
        box-shadow: 0 0 12px rgba(251, 191, 36, 0.20);
    }}
    .pill-high {{
        background: rgba(251, 113, 133, 0.18);
        color: {c["coral"]};
        border: 1px solid rgba(251, 113, 133, 0.40);
        box-shadow: 0 0 14px rgba(251, 113, 133, 0.25);
        animation: pulse-coral 2s infinite ease-in-out;
    }}
    .pill-hyper {{
        background: rgba(225, 29, 72, {'0.28' if is_dark else '0.12'});
        color: {'#FDA4AF' if is_dark else '#9F1239'};
        border: 1px solid rgba(225, 29, 72, 0.60);
        box-shadow: 0 0 18px rgba(225, 29, 72, 0.40);
        animation: pulse-coral 1.5s infinite ease-in-out;
    }}

    @keyframes pulse-coral {{
        0% {{ box-shadow: 0 0 0 0 rgba(251, 113, 133, 0.4); }}
        70% {{ box-shadow: 0 0 0 10px rgba(251, 113, 133, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(251, 113, 133, 0); }}
    }}

    /* Glassmorphism KPI Card */
    .glass-card {{
        background: {c["surface_card"]};
        border: 1px solid {c["surface_card_border"]};
        border-radius: 16px;
        padding: 16px 18px;
        backdrop-filter: blur(12px);
        box-shadow: {'0 6px 20px rgba(0, 0, 0, 0.30)' if is_dark else '0 2px 12px rgba(15, 23, 42, 0.05)'};
        transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.2s ease, border-color 0.2s ease;
        margin-bottom: 12px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 124px;
    }}
    .glass-card:hover {{
        transform: translateY(-3px);
        border-color: {c["primary"]}55;
        box-shadow: {'0 12px 28px rgba(0, 0, 0, 0.45)' if is_dark else '0 6px 20px rgba(15, 23, 42, 0.09)'};
    }}

    .kpi-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}
    .kpi-label {{
        font-size: 0.76rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {c["text_muted"]};
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .kpi-val-row {{
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        margin: 6px 0 2px 0;
    }}
    .kpi-hero-num {{
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 2.1rem;
        font-weight: 800;
        line-height: 1.1;
    }}
    .kpi-standard-num {{
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 1.65rem;
        font-weight: 700;
        line-height: 1.1;
        color: {c["text_primary"]};
    }}
    .kpi-footer {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.74rem;
        color: {c["text_subtle"]};
    }}

    /* Compact Wrapped Clinical Profile Chips */
    .chip {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: {'rgba(30, 41, 59, 0.65)' if is_dark else '#E2E8F0'};
        border: 1px solid {c["surface_card_border"]};
        color: {c["text_primary"]};
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.76rem;
        font-weight: 500;
        margin: 3px 2px;
        transition: background 0.15s ease;
    }}
    .chip-dx {{
        background: {'rgba(139, 92, 246, 0.15)' if is_dark else 'rgba(124, 58, 237, 0.12)'};
        border-color: {'rgba(139, 92, 246, 0.35)' if is_dark else 'rgba(124, 58, 237, 0.30)'};
        color: {c["violet"]};
    }}
    .chip-rx {{
        background: {'rgba(56, 189, 248, 0.15)' if is_dark else 'rgba(2, 132, 199, 0.12)'};
        border-color: {'rgba(56, 189, 248, 0.35)' if is_dark else 'rgba(2, 132, 199, 0.30)'};
        color: {c["cyan"]};
    }}

    /* Twin Insight Card */
    .insight-card {{
        background: {'linear-gradient(135deg, rgba(45, 212, 191, 0.08) 0%, rgba(139, 92, 246, 0.08) 100%)' if is_dark else 'linear-gradient(135deg, rgba(13, 148, 136, 0.06) 0%, rgba(124, 58, 237, 0.05) 100%)'};
        border: 1px solid {'rgba(45, 212, 191, 0.25)' if is_dark else 'rgba(13, 148, 136, 0.20)'};
        border-left: 4px solid {c["primary"]};
        border-radius: 12px;
        padding: 14px 18px;
        margin: 12px 0 16px 0;
        color: {c["text_primary"]};
        font-size: 0.90rem;
        line-height: 1.5;
    }}

    /* Clean Styled Primary Teal Buttons */
    .stButton>button[kind="primary"] {{
        background: linear-gradient(135deg, {c["primary"]} 0%, {c["cyan"]} 100%) !important;
        color: #0B1220 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 8px 20px !important;
        box-shadow: 0 4px 14px {c["primary"]}33 !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }}
    .stButton>button[kind="primary"]:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px {c["primary"]}55 !important;
    }}

    /* Tab Styling & Smooth Fade-in */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        border-bottom: 1px solid {c["surface_card_border"]};
        padding-bottom: 2px;
    }}
    /* Older Streamlit renders tabs as data-baseweb="tab", newer as data-testid="stTab" */
    .stTabs [data-baseweb="tab"], .stTabs [data-testid="stTab"] {{
        padding: 8px 18px;
        font-weight: 600;
        font-size: 0.90rem;
        color: {c["text_muted"]} !important;
        border-radius: 8px 8px 0 0;
        transition: all 0.2s ease;
    }}
    .stTabs [data-testid="stTab"] p {{ color: inherit !important; }}
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {{
        color: var(--muted) !important;
    }}
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p {{
        color: var(--text) !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: {'rgba(45, 212, 191, 0.12)' if is_dark else 'rgba(13, 148, 136, 0.10)'} !important;
        color: {c["primary"]} !important;
        border-bottom: 2px solid {c["primary"]} !important;
    }}
    div[data-testid="stTabContent"] {{
        animation: tabFadeIn 0.22s ease-in-out;
    }}
    @keyframes tabFadeIn {{
        from {{ opacity: 0.2; transform: translateY(3px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* Consistent Micro-label */
    .micro-label {{
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }}

    /* Upload Progress Stepper */
    .stepper-container {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 12px 24px;
        margin-bottom: 20px;
    }}
    .step-item {{
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 0.80rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--muted);
    }}
    .step-item.active {{
        color: var(--accent);
    }}
    .step-item.completed {{
        color: var(--accent);
    }}
    .step-num {{
        display: flex;
        align-items: center;
        justify-content: center;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background: var(--border);
        color: var(--text);
        font-size: 0.75rem;
        font-weight: 800;
    }}
    .step-item.active .step-num,
    .step-item.completed .step-num {{
        background: var(--accent);
        color: #FFFFFF;
        box-shadow: 0 0 10px var(--accent)66;
    }}
    .step-divider {{
        flex: 1;
        height: 2px;
        background: var(--border);
        margin: 0 14px;
    }}
    .step-divider.active {{
        background: var(--accent);
    }}

    /* Sliders and Inputs */
    div[data-testid="stSlider"] {{
        padding-top: 8px;
        padding-bottom: 8px;
    }}
    input, textarea, [data-baseweb="input"] {{
        background-color: var(--surface) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
    }}

    /* File Uploader Dropzone */
    [data-testid="stFileUploader"] {{
        background: var(--surface) !important;
        border: 2px dashed var(--accent) !important;
        border-radius: 14px !important;
        padding: 16px !important;
        box-shadow: {'0 4px 18px rgba(0,0,0,0.25)' if is_dark else '0 2px 10px rgba(15,23,42,0.04)'};
    }}
    [data-testid="stFileUploader"] * {{
        color: var(--text) !important;
    }}
    [data-testid="stFileUploaderDropzone"] {{
        background: transparent !important;
    }}

    /* Expanders & DataFrames & Alerts */
    .streamlit-expanderHeader {{
        background: var(--surface) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }}
    div[data-testid="stDataFrame"] {{
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }}
    div[data-testid="stAlert"] {{
        background: var(--surface) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }}

    /* Friendly Empty State & Error Cards */
    .empty-state-card {{
        background: var(--surface-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 36px 24px;
        text-align: center;
        margin: 20px 0;
        box-shadow: {'0 6px 20px rgba(0, 0, 0, 0.25)' if is_dark else '0 2px 12px rgba(15, 23, 42, 0.05)'};
    }}
    .error-card {{
        background: var(--surface-card);
        border: 1px solid var(--coral);
        border-left: 5px solid var(--coral);
        border-radius: 12px;
        padding: 18px 22px;
        margin: 16px 0;
    }}
    </style>
    """


# ══════════════════════════════════════════════════════════════════════════════
# Stylized Human Body "Digital Twin" SVG Generator (HTML Component)
# ══════════════════════════════════════════════════════════════════════════════

def render_digital_twin_html(
    glucose_val: float,
    hr_val: float,
    hrv_val: float,
    step_val: int,
    sleep_val: float,
    theme_mode: str = "dark",
) -> str:
    """Render an interactive futuristic anatomical Digital Twin SVG with glowing data beacons."""
    c = THEME_COLORS.get(theme_mode, THEME_COLORS["dark"])

    # Determine node colors based on clinical states
    glu_color = c["green"] if (70 <= glucose_val <= 180) else (c["coral"] if glucose_val > 180 else c["amber"])
    hr_color = c["coral"] if hr_val > 100 else (c["amber"] if hr_val > 85 else c["green"])
    sleep_color = c["violet"]
    activity_color = c["primary"]

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
      body {{
        margin: 0;
        padding: 0;
        background: transparent;
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif;
        overflow: hidden;
      }}
      .svg-wrap {{
        width: 100%;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
      }}
      .beacon {{
        cursor: pointer;
        transition: transform 0.2s ease;
      }}
      .beacon:hover {{
        transform: scale(1.25);
      }}
      .beacon:hover .node-card {{
        opacity: 1;
        visibility: visible;
      }}
      .pulse-ring {{
        animation: ring-pulse 2.2s infinite cubic-bezier(0.25, 1, 0.5, 1);
        transform-origin: center;
      }}
      @keyframes ring-pulse {{
        0% {{ r: 5px; opacity: 0.9; }}
        100% {{ r: 18px; opacity: 0; }}
      }}
      .conduit {{
        stroke-dasharray: 4, 4;
        animation: conduit-flow 3s linear infinite;
      }}
      @keyframes conduit-flow {{
        from {{ stroke-dashoffset: 16; }}
        to {{ stroke-dashoffset: 0; }}
      }}
      .node-card {{
        opacity: 0;
        visibility: hidden;
        transition: opacity 0.2s ease, visibility 0.2s ease;
        pointer-events: none;
      }}
    </style>
    </head>
    <body>
      <div class="svg-wrap">
        <svg width="260" height="150" viewBox="0 0 260 150" fill="none">
          <!-- Ambient Glow Core -->
          <circle cx="130" cy="75" r="45" fill="url(#coreGlow)" opacity="0.35"/>
          
          <defs>
            <radialGradient id="coreGlow" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stop-color="{c["primary"]}" stop-opacity="0.4"/>
              <stop offset="100%" stop-color="{c["primary"]}" stop-opacity="0"/>
            </radialGradient>
            <filter id="glowFilter" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="3" result="blur"/>
              <feMerge>
                <feMergeNode in="blur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
          </defs>

          <!-- Stylized Human Wireframe Silhouette -->
          <g stroke="{c["surface_card_border"]}" stroke-width="1.2" opacity="0.75" fill="none">
            <!-- Head -->
            <ellipse cx="130" cy="24" rx="11" ry="13"/>
            <!-- Neck -->
            <line x1="130" y1="37" x2="130" y2="43"/>
            <!-- Clavicle & Shoulders -->
            <path d="M106 48 C 118 45, 142 45, 154 48"/>
            <!-- Torso / Spine axis -->
            <line x1="130" y1="43" x2="130" y2="92" stroke="{c["primary"]}" stroke-width="1.4" opacity="0.5"/>
            <!-- Ribcage contour -->
            <path d="M112 55 C 112 72, 148 72, 148 55"/>
            <path d="M115 68 C 115 84, 145 84, 145 68"/>
            <!-- Pelvis -->
            <path d="M116 92 C 124 97, 136 97, 144 92"/>
            <!-- Arms -->
            <path d="M106 48 L 98 75 L 94 98"/>
            <path d="M154 48 L 162 75 L 166 98"/>
            <!-- Legs -->
            <path d="M122 95 L 118 124 L 116 144"/>
            <path d="M138 95 L 142 124 L 144 144"/>
          </g>

          <!-- Neural / Physiological Conduits -->
          <path class="conduit" d="M130 24 Q 105 40 122 60" stroke="{c["violet"]}" stroke-width="1" opacity="0.6"/>
          <path class="conduit" d="M122 60 Q 140 70 130 78" stroke="{glu_color}" stroke-width="1.2" opacity="0.8"/>
          <path class="conduit" d="M130 78 Q 110 110 118 135" stroke="{c["primary"]}" stroke-width="1" opacity="0.6"/>

          <!-- 1. SLEEP / CIRCADIAN NODE (Brain / Head) -->
          <g class="beacon" transform="translate(130, 24)">
            <circle class="pulse-ring" cx="0" cy="0" r="5" stroke="{sleep_color}" stroke-width="1.2" fill="none"/>
            <circle cx="0" cy="0" r="4.5" fill="{sleep_color}" filter="url(#glowFilter)"/>
            <!-- Tooltip Card -->
            <g class="node-card" transform="translate(14, -18)">
              <rect x="0" y="0" width="105" height="28" rx="6" fill="{c["surface"]}" stroke="{sleep_color}" stroke-width="1"/>
              <text x="7" y="12" fill="{c["text_muted"]}" font-size="8" font-weight="600">SLEEP / CIRCADIAN</text>
              <text x="7" y="23" fill="{c["text_primary"]}" font-size="10" font-weight="700">{sleep_val:.1f}h Rest</text>
            </g>
          </g>

          <!-- 2. HEART / HRV NODE (Left Chest) -->
          <g class="beacon" transform="translate(123, 58)">
            <circle class="pulse-ring" cx="0" cy="0" r="5" stroke="{hr_color}" stroke-width="1.2" fill="none"/>
            <circle cx="0" cy="0" r="5" fill="{hr_color}" filter="url(#glowFilter)"/>
            <!-- Tooltip Card -->
            <g class="node-card" transform="translate(-115, -12)">
              <rect x="0" y="0" width="105" height="28" rx="6" fill="{c["surface"]}" stroke="{hr_color}" stroke-width="1"/>
              <text x="7" y="12" fill="{c["text_muted"]}" font-size="8" font-weight="600">HEART & HRV</text>
              <text x="7" y="23" fill="{c["text_primary"]}" font-size="10" font-weight="700">{hr_val:.0f} bpm • {hrv_val:.0f} ms</text>
            </g>
          </g>

          <!-- 3. PANCREAS / GLUCOSE NODE (Mid-Torso) -->
          <g class="beacon" transform="translate(130, 78)">
            <circle class="pulse-ring" cx="0" cy="0" r="6" stroke="{glu_color}" stroke-width="1.5" fill="none"/>
            <circle cx="0" cy="0" r="6" fill="{glu_color}" filter="url(#glowFilter)"/>
            <!-- Tooltip Card -->
            <g class="node-card" transform="translate(16, -14)">
              <rect x="0" y="0" width="110" height="28" rx="6" fill="{c["surface"]}" stroke="{glu_color}" stroke-width="1.2"/>
              <text x="7" y="12" fill="{c["text_muted"]}" font-size="8" font-weight="600">PANCREAS & CGM</text>
              <text x="7" y="23" fill="{glu_color}" font-size="10.5" font-weight="800">{glucose_val:.0f} mg/dL</text>
            </g>
          </g>

          <!-- 4. ACTIVITY / METABOLIC NODE (Lower Limb) -->
          <g class="beacon" transform="translate(142, 126)">
            <circle class="pulse-ring" cx="0" cy="0" r="5" stroke="{activity_color}" stroke-width="1.2" fill="none"/>
            <circle cx="0" cy="0" r="4.5" fill="{activity_color}" filter="url(#glowFilter)"/>
            <!-- Tooltip Card -->
            <g class="node-card" transform="translate(-115, -10)">
              <rect x="0" y="0" width="105" height="28" rx="6" fill="{c["surface"]}" stroke="{activity_color}" stroke-width="1"/>
              <text x="7" y="12" fill="{c["text_muted"]}" font-size="8" font-weight="600">METABOLIC STEPS</text>
              <text x="7" y="23" fill="{activity_color}" font-size="10" font-weight="700">{step_val:,} steps</text>
            </g>
          </g>
        </svg>
      </div>
    </body>
    </html>
    """


# ══════════════════════════════════════════════════════════════════════════════
# Shared Plotly Layout Template & apply_theme Helper
# ══════════════════════════════════════════════════════════════════════════════

def get_shared_plotly_layout(theme_mode: str = "dark", height: int = 400) -> Dict[str, Any]:
    c = THEME_COLORS.get(theme_mode, THEME_COLORS["dark"])
    return dict(
        font=dict(family="'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif", color=c["text_primary"], size=11),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=c["plot_bg"],
        height=height,
        margin=dict(l=45, r=45, t=40, b=55),
        xaxis=dict(
            gridcolor=c["grid"],
            zerolinecolor=c["grid"],
            linecolor=c["surface_card_border"],
            tickfont=dict(color=c["text_muted"], size=10),
            title_font=dict(color=c["text_primary"], size=11),
        ),
        yaxis=dict(
            gridcolor=c["grid"],
            zerolinecolor=c["grid"],
            linecolor=c["surface_card_border"],
            tickfont=dict(color=c["text_muted"], size=10),
            title_font=dict(color=c["text_primary"], size=11),
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="left",
            x=0,
            font=dict(size=10, color=c["text_muted"]),
        ),
        hoverlabel=dict(
            bgcolor=c["surface"],
            font_color=c["text_primary"],
            bordercolor=c["surface_card_border"],
        ),
    )


def apply_theme(fig: Any, theme_mode: str = "dark", height: Optional[int] = None) -> Any:
    """Apply unified health-tech styling to any Plotly figure, guaranteeing WCAG AA contrast."""
    c = THEME_COLORS.get(theme_mode, THEME_COLORS["dark"])
    is_dark = (theme_mode == "dark")
    layout_update: Dict[str, Any] = dict(
        font=dict(family="'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif", color=c["text_primary"], size=11),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=c["plot_bg"],
        hoverlabel=dict(
            bgcolor=c["surface"],
            font_color=c["text_primary"],
            bordercolor=c["surface_card_border"],
            font_size=11,
            font_family="'Inter', sans-serif",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=10, color=c["text_muted"]),
        ),
        margin=dict(l=45, r=45, t=45, b=40),
    )
    if height is not None:
        layout_update["height"] = height

    fig.update_layout(**layout_update)
    fig.update_xaxes(
        gridcolor=c["grid"],
        zerolinecolor=c["grid"],
        linecolor=c["surface_card_border"],
        tickfont=dict(color=c["text_muted"], size=10),
        title_font=dict(color=c["text_primary"], size=11),
    )
    fig.update_yaxes(
        gridcolor=c["grid"],
        zerolinecolor=c["grid"],
        linecolor=c["surface_card_border"],
        tickfont=dict(color=c["text_muted"], size=10),
        title_font=dict(color=c["text_primary"], size=11),
    )
    return fig

