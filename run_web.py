import os
import sys
import socket
import webbrowser
import threading
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def open_browser():
    time.sleep(1.0)
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    print("=" * 65)
    print(" [InsurAI] Enterprise Full-Stack Platform")
    print(" Web Portal  : http://127.0.0.1:8000")
    print(" REST API Doc: http://127.0.0.1:8000/docs")
    print("=" * 65)

    if is_port_in_use(8000):
        print("[OK] Server is already active on port 8000! Opening browser...")
        open_browser()
    else:
        import uvicorn
        from server import app
        threading.Thread(target=open_browser, daemon=True).start()
        uvicorn.run(app, host="127.0.0.1", port=8000)
