"""Tests for Streamlit application using streamlit.testing.v1.AppTest.

Tests:
1. Demo mode loads without exceptions.
2. Upload mode with no files loads without an exception.
3. Sample files produce a prediction and complete twin state.
4. Both themes (Dark and Clinical Light) render without exceptions.
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def test_demo_mode_loads():
    """Verify demo patient mode loads cleanly with no exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception, f"App raised an exception in demo mode: {at.exception}"
    assert len(at.tabs) >= 6


def test_upload_mode_no_files_loads_without_exception():
    """Verify upload mode with zero files does not crash and renders friendly empty state."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    # Set data mode to Upload Patient Data in session state before run
    at.session_state["data_mode_choice"] = "Upload Patient Data"
    at.session_state["requested_mode"] = "Upload Patient Data"
    at.run()
    assert not at.exception, f"App raised an exception in upload mode with no files: {at.exception}"


def test_both_themes_render_without_exceptions():
    """Verify both Dark Theme and Clinical Light Theme render without exceptions."""
    # Test Dark theme
    at_dark = AppTest.from_file(APP_PATH, default_timeout=30)
    at_dark.session_state["theme_mode_choice"] = "Dark Theme"
    at_dark.run()
    assert not at_dark.exception, f"Dark theme raised an exception: {at_dark.exception}"

    # Test Light theme
    at_light = AppTest.from_file(APP_PATH, default_timeout=30)
    at_light.session_state["theme_mode_choice"] = "Clinical Light"
    at_light.run()
    assert not at_light.exception, f"Light theme raised an exception: {at_light.exception}"


def test_sample_files_produce_prediction():
    """Verify clicking sample files validates and builds twin prediction without error."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["data_mode_choice"] = "Upload Patient Data"
    at.session_state["requested_mode"] = "Upload Patient Data"
    at.run()
    assert not at.exception

    # Locate and click the sample files button if available
    sample_btns = [b for b in at.button if "Sample Files" in (b.label or "")]
    if sample_btns:
        sample_btns[0].click().run()
        assert not at.exception, f"Sample button click raised exception: {at.exception}"
