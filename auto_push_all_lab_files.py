import os
import time
import subprocess

# Path to your Git folder
WATCH_FOLDER = r"C:\Users\Udhaya\sem5_ML"

# Time interval in seconds (checks every 30 seconds)
CHECK_INTERVAL = 30

# Keep track of last modified time for each file
last_known_times = {}

def has_any_py_change():
    global last_known_times
    changed = False

    for root, dirs, files in os.walk(WATCH_FOLDER):
        for file in files:
            if file.endswith((".py", ".md")):

                full_path = os.path.join(root, file)
                mod_time = os.path.getmtime(full_path)

                if full_path not in last_known_times or last_known_times[full_path] != mod_time:
                    last_known_times[full_path] = mod_time
                    changed = True
    return changed

while True:
    if has_any_py_change():
        print(" Detected change in .py files. Auto-pushing to GitHub...")

        try:
            subprocess.run(["git", "add", "."], cwd=WATCH_FOLDER)
            subprocess.run(["git", "commit", "-m", "Auto-push new or updated Python file"], cwd=WATCH_FOLDER)
            subprocess.run(["git", "push"], cwd=WATCH_FOLDER)
            print(" Pushed successfully.")
        except Exception as e:
            print(" Error during push:", e)

    time.sleep(CHECK_INTERVAL)
