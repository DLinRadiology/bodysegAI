import os
import sys
import socket
import webbrowser
import threading

PORT = 5005
URL = f"http://127.0.0.1:{PORT}"

# When frozen by PyInstaller, resolve paths relative to the bundle dir
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
    os.environ['BODYSEGAI_BASE'] = BASE_DIR
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    os.environ['BODYSEGAI_BASE'] = BASE_DIR


def is_already_running():
    """Check if another instance is already serving on our port."""
    try:
        s = socket.create_connection(("127.0.0.1", PORT), timeout=1)
        s.close()
        return True
    except (ConnectionRefusedError, OSError, socket.timeout):
        return False


def open_browser():
    webbrowser.open(URL)


if __name__ == "__main__":
    if is_already_running():
        print(f"BodySegAI is already running at {URL}")
        open_browser()
        sys.exit(0)

    from bodysegai.app import app

    threading.Timer(1.5, open_browser).start()
    app.run(host="127.0.0.1", port=PORT, debug=False)
