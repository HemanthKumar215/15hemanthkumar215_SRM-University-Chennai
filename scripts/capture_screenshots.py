"""Automated screenshot capture script for GlycoTwin.

Captures all tabs in Dark Theme and Clinical Light Theme at 1920x1080 and 1366x768.

Usage (with the dashboard running):
    python scripts/capture_screenshots.py [--url http://localhost:8501] [--tabs explainability whatif]
                                          [--patient "Aditya Patel"]
"""

import argparse
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
    ("Dark Midnight", "dark"),
    ("Clinical Light", "light"),
]


def wait_for_app(page, timeout_ms: int = 120_000) -> None:
    """Block until Streamlit has finished its script run and charts have drawn.

    Screenshots taken earlier show an empty frame (the previous version of this
    script captured every image before the app had rendered any content).
    """
    page.wait_for_selector('[data-testid="stTab"]', timeout=timeout_ms)
    # The status widget ("Running...") is present while the script executes
    page.wait_for_function(
        """() => {
            const s = document.querySelector('[data-testid="stStatusWidget"]');
            return !s || !s.innerText.includes('Running');
        }""",
        timeout=timeout_ms,
    )
    time.sleep(2.5)  # Plotly animations / fonts


def select_patient(page, name: str) -> None:
    """Pick a demo patient by (partial) name in the sidebar selector."""
    box = page.locator('[data-testid="stSelectbox"]').filter(has_text="Select Patient Twin")
    box.locator("input").first.click()
    page.keyboard.type(name)
    page.keyboard.press("Enter")
    time.sleep(1)
    wait_for_app(page)


def run_capture(url: str = "http://localhost:8501", only_tabs: list | None = None,
                patient: str | None = None):
    print("Starting automated screenshot capture...")
    tabs = [t for t in TABS if not only_tabs or t[1] in only_tabs]
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=EDGE_PATH if Path(EDGE_PATH).exists() else None,
                                    headless=True)

        for width, height, res_label in VIEWPORTS:
            for theme_label, theme_name in THEMES:
                context = browser.new_context(viewport={"width": width, "height": height})
                page = context.new_page()
                page.goto(url, wait_until="networkidle")
                wait_for_app(page)
                if patient:
                    select_patient(page, patient)

                # Switch theme in sidebar if needed
                try:
                    theme_radio = page.locator(f'label:has-text("{theme_label}")')
                    if theme_radio.count() > 0:
                        theme_radio.first.click()
                        time.sleep(1)
                        wait_for_app(page)
                except Exception as e:
                    print(f"Warning toggling theme {theme_label}: {e}")

                for tab_label, tab_name in tabs:
                    try:
                        page.get_by_role("tab", name=tab_label).first.click()
                        time.sleep(2.5)  # allow charts in the newly shown tab to render

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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8501", help="Running dashboard URL")
    parser.add_argument("--tabs", nargs="*", choices=[t[1] for t in TABS],
                        help="Only capture these tabs (default: all)")
    parser.add_argument("--patient", help="Demo patient to show, by name (default: the app's first patient)")
    args = parser.parse_args()
    run_capture(args.url, args.tabs, args.patient)
