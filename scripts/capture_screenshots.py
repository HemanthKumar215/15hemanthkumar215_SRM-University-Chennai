"""Automated screenshot capture script for GlycoTwin.

Captures all tabs in Dark Theme and Clinical Light Theme at 1920x1080 and 1366x768.
"""

from pathlib import Path
import time
from playwright.sync_api import sync_playwright

DOCS_IMG = Path(__file__).resolve().parent.parent / "docs" / "img"
DOCS_IMG.mkdir(parents=True, exist_ok=True)

EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

TABS = [
    ("Overview", "overview"),
    ("Glucose & Vitals", "glucose_vitals"),
    ("What-If Simulator", "whatif"),
    ("Explainability", "explainability"),
    ("Model Report", "model_report"),
    ("Upload Data", "upload"),
]

VIEWPORTS = [
    (1920, 1080, "1920"),
    (1366, 768, "1366"),
]

THEMES = [
    ("Dark Theme", "dark"),
    ("Clinical Light", "light"),
]


def run_capture():
    print("Starting automated screenshot capture...")
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=EDGE_PATH, headless=True)

        for width, height, res_label in VIEWPORTS:
            for theme_label, theme_name in THEMES:
                context = browser.new_context(viewport={"width": width, "height": height})
                page = context.new_page()
                page.goto("http://localhost:8501", wait_until="networkidle")
                time.sleep(3)  # wait for initial render

                # Switch theme in sidebar if needed
                try:
                    theme_radio = page.locator(f'label:has-text("{theme_label}")')
                    if theme_radio.count() > 0:
                        theme_radio.first.click()
                        time.sleep(2)
                except Exception as e:
                    print(f"Warning toggling theme {theme_label}: {e}")

                for tab_label, tab_name in TABS:
                    try:
                        tab_btn = page.locator(f'[data-baseweb="tab"]:has-text("{tab_label}")')
                        if tab_btn.count() > 0:
                            tab_btn.first.click()
                            time.sleep(2)  # allow charts to render

                        filename = f"dashboard_{res_label}_{theme_name}_{tab_name}.png"
                        out_path = DOCS_IMG / filename
                        page.screenshot(path=str(out_path), full_page=False)
                        print(f"Saved: {filename}")

                        # Also save standard filenames for README
                        if res_label == "1920" and theme_name == "dark":
                            std_path = DOCS_IMG / f"dashboard_1920_{tab_name}.png"
                            page.screenshot(path=str(std_path), full_page=False)
                        elif res_label == "1366" and theme_name == "dark":
                            std_path = DOCS_IMG / f"dashboard_1366_{tab_name}.png"
                            page.screenshot(path=str(std_path), full_page=False)
                        elif tab_name == "overview" and theme_name == "light":
                            std_path = DOCS_IMG / f"dashboard_{res_label}_light_theme.png"
                            page.screenshot(path=str(std_path), full_page=False)
                    except Exception as e:
                        print(f"Error capturing {res_label} {theme_name} {tab_name}: {e}")

                context.close()

        browser.close()
    print("Screenshot capture complete!")


if __name__ == "__main__":
    run_capture()
