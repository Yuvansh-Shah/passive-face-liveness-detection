"""Khoji liveness demo UI.

Serves the existing FastAPI app from api.py (unchanged) plus a static
web frontend on the same port.

Run:   python demo.py           (from the liveness/ folder, or anywhere)
Open:  http://127.0.0.1:8000    (opens automatically; pass --no-browser to skip)
"""

import socket
import sys
import threading
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # so `import api` works from any cwd

import uvicorn
from fastapi.staticfiles import StaticFiles
from PIL import Image

# allow ultra-high-res phone photos (e.g. 200MP camera modes) that exceed
# PIL's default decompression-bomb threshold; fine for a localhost demo
Image.MAX_IMAGE_PIXELS = 300_000_000

from api import app  # loads the model once, exactly as `python api.py` does

app.mount("/", StaticFiles(directory=HERE / "static", html=True), name="ui")

HOST, PORT = "127.0.0.1", 8000
URL = f"http://{HOST}:{PORT}"


def port_is_free() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((HOST, PORT))
            return True
        except OSError:
            return False


if __name__ == "__main__":
    if not port_is_free():
        sys.exit(
            f"ERROR: port {PORT} is already in use.\n"
            f"Another server (an old `python api.py` or `python demo.py`) is "
            f"probably still running - close that window/process and rerun."
        )
    if "--no-browser" not in sys.argv:
        # model is already loaded by the import above, so the server is up
        # well within a second of uvicorn starting
        threading.Timer(1.5, webbrowser.open, args=(URL,)).start()
    print(f"Khoji liveness demo: {URL}")
    uvicorn.run(app, host=HOST, port=PORT)
