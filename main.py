import os
import sys
import socket
import webbrowser
import threading
import urllib.parse

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


# Loading page shown instantly while the server starts up
_LOADING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>BodySegAI — Starting...</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    display: flex; align-items: center; justify-content: center;
    height: 100vh; background: #0F172A; color: #E2E8F0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    flex-direction: column; gap: 32px;
  }
  .logo {
    width: 72px; height: 72px; border-radius: 16px;
    background: linear-gradient(135deg, #3B82F6, #8B5CF6);
    display: flex; align-items: center; justify-content: center;
  }
  .logo svg { width: 40px; height: 40px; color: white; }
  h1 { font-size: 28px; font-weight: 700; color: #F8FAFC; }
  .bar-track {
    width: 280px; height: 4px; background: #1E293B;
    border-radius: 2px; overflow: hidden;
  }
  .bar-fill {
    width: 30%; height: 100%;
    background: linear-gradient(90deg, #3B82F6, #8B5CF6);
    border-radius: 2px;
    animation: loading 1.2s ease-in-out infinite;
  }
  @keyframes loading {
    0%   { transform: translateX(-100%); }
    100% { transform: translateX(400%); }
  }
  .status { font-size: 14px; color: #64748B; }
</style>
</head>
<body>
  <div class="logo">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M12 2a5 5 0 0 1 5 5v3a5 5 0 0 1-10 0V7a5 5 0 0 1 5-5z"/>
      <path d="M8 14s-3 2-3 5a3 3 0 0 0 3 3h8a3 3 0 0 0 3-3c0-3-3-5-3-5"/>
    </svg>
  </div>
  <h1>BodySegAI</h1>
  <div class="bar-track"><div class="bar-fill"></div></div>
  <div class="status">Loading components...</div>
  <script>
    const url = "SERVER_URL";
    (async function poll() {
      try {
        const r = await fetch(url, { mode: "no-cors" });
        window.location.replace(url);
      } catch (e) {
        setTimeout(poll, 500);
      }
    })();
  </script>
</body>
</html>"""


def open_loading_page():
    """Open browser immediately with a loading page that redirects when server is ready."""
    html = _LOADING_HTML.replace("SERVER_URL", URL)
    data_uri = "data:text/html;charset=utf-8," + urllib.parse.quote(html)
    webbrowser.open(data_uri)


if __name__ == "__main__":
    if is_already_running():
        print(f"BodySegAI is already running at {URL}")
        webbrowser.open(URL)
        sys.exit(0)

    # Show loading page immediately, before heavy imports
    open_loading_page()

    from bodysegai.app import app

    app.run(host="127.0.0.1", port=PORT, debug=False)
