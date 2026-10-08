# =============================================================================
# serve.py
# Maternal & Infant Mortality Risk Screening — Modern Web Server
# Serves the Glassmorphic / Claymorphic HTML5 web application on localhost
# =============================================================================

import os
import sys
import json
import webbrowser
import http.server
import socketserver

PORT = 8000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, 'web')

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def end_headers(self):
        # Enable CORS and disable aggressive caching for local testing
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

def main():
    os.chdir(WEB_DIR)
    url = f"http://localhost:{PORT}"
    print(f"\n" + "=" * 70)
    print(f"🤰 Maternal & Infant Mortality Risk AI — Web Platform")
    print(f"✨ Dual Morphism UI: Glassmorphism & Claymorphism")
    print(f"🌐 Serving at: {url}")
    print(f"=" * 70 + "\n")

    try:
        with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
            print(f"Server online. Press Ctrl+C to stop.")
            try:
                webbrowser.open(url)
            except Exception:
                pass
            httpd.serve_forever()
    except OSError as e:
        if "address already in use" in str(e).lower() or e.errno == 98 or e.errno == 10048:
            alt_port = 8080
            alt_url = f"http://localhost:{alt_port}"
            print(f"Port {PORT} in use, trying {alt_port}...")
            with socketserver.TCPServer(("", alt_port), CustomHandler) as httpd:
                print(f"🌐 Serving at: {alt_url}")
                try:
                    webbrowser.open(alt_url)
                except Exception:
                    pass
                httpd.serve_forever()
        else:
            raise e

if __name__ == '__main__':
    main()
