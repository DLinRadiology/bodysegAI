import os
import sys
import webbrowser
import threading

# When frozen by PyInstaller, resolve paths relative to the bundle dir
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
    os.environ['BODYSEGAI_BASE'] = BASE_DIR
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    os.environ['BODYSEGAI_BASE'] = BASE_DIR

from bodysegai.app import app

PORT = 5005

def open_browser():
    webbrowser.open(f"http://127.0.0.1:{PORT}")

if __name__ == "__main__":
    threading.Timer(1.5, open_browser).start()
    app.run(host="127.0.0.1", port=PORT, debug=False)
