import os
import sys

print("Python :", sys.executable)
print("Dossier :", os.getcwd())
print("PID :", os.getpid())

from app.main import app

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=8050,
        debug=True,
        use_reloader=True,
        dev_tools_hot_reload=True,
        dev_tools_hot_reload_interval=1,
        dev_tools_hot_reload_watch_interval=0.5,
    )