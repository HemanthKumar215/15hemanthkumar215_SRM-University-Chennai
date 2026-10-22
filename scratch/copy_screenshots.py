import os
import glob
import shutil

src_dir = r"C:\Users\P Hemanth Kumar\.gemini\antigravity-ide\brain\25fe6bd5-7f3e-4735-8da5-9ca3c4ca5eb7"
dest_dir = r"c:\Users\P Hemanth Kumar\digital twin\docs\img"
os.makedirs(dest_dir, exist_ok=True)

targets = {
    'dashboard_1920_overview.png': 'dashboard_1920_overview_17*.png',
    'dashboard_1920_overview_charts.png': 'dashboard_1920_overview_charts_17*.png',
    'dashboard_1920_glucose_vitals.png': 'dashboard_1920_glucose_vitals*.png',
    'dashboard_1920_whatif.png': 'dashboard_1920_whatif_17*.png',
    'dashboard_1920_explainability.png': 'tab_explainability_1920*.png',
    'dashboard_1920_model_report.png': 'tab_model_report_1920*.png',
    'dashboard_1920_upload.png': 'tab_upload_data_1920*.png',
    'dashboard_1920_light_theme.png': 'dashboard_1920_light_theme_17*.png',
    'dashboard_1366_overview.png': 'dashboard_1366_overview_17*.png',
    'dashboard_1366_glucose_vitals.png': 'dashboard_1366_glucose_vitals*.png',
    'dashboard_1366_whatif.png': '*whatif*1366*.png',
    'dashboard_1366_explainability.png': 'dashboard_1366_explainability*.png',
    'dashboard_1366_model_report.png': 'dashboard_1366_model_report*.png',
    'dashboard_1366_upload.png': 'dashboard_1366_upload*.png',
    'dashboard_1366_light_theme.png': 'step7_theme_light_1366*.png',
}

for dest_name, pattern in targets.items():
    matches = glob.glob(os.path.join(src_dir, pattern))
    if not matches:
        # Check subdirectories
        matches = glob.glob(os.path.join(src_dir, "**", pattern), recursive=True)
    if matches:
        # Sort by mtime descending
        matches.sort(key=os.path.getmtime, reverse=True)
        best = matches[0]
        dest_path = os.path.join(dest_dir, dest_name)
        shutil.copy2(best, dest_path)
        print(f"Copied {os.path.basename(best)} -> {dest_name} ({os.path.getsize(dest_path)} bytes)")
    else:
        print(f"Warning: No match found for {pattern}")

print("\nFinal docs/img directory:")
for f in sorted(os.listdir(dest_dir)):
    sz = os.path.getsize(os.path.join(dest_dir, f))
    print(f"  {f}: {sz} bytes")
