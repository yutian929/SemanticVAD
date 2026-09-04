r"""
Static file server for the SoulX-Duplug web client.

This only serves the `web/` directory. It does NOT touch the inference server
(`server.py`) — the page talks to `ws://<host>:8000/turn` directly from the
browser, using exactly the same protocol as `test.py`.

Why a tiny server instead of opening index.html directly?
  * `getUserMedia` requires a secure context: http://localhost or HTTPS.
    Opening the file via file:// is not reliable (Safari refuses it).
  * `AudioWorklet.addModule()` cannot load modules over file:// (CORS).

No dependencies: standard library only, so no virtualenv is needed for this
script (`install_client.sh` is only for the Python clients `test.py` etc.).

Usage:
    python web_client.py                    # http://127.0.0.1:8080
    python web_client.py --port 9000
    python web_client.py --host 0.0.0.0     # LAN (needs HTTPS for microphone)
    python web_client.py --no-browser
    python web_client.py --ws ws://127.0.0.1:8000/turn   # prefill the ws url


Case 1 — inference server on the same machine
---------------------------------------------
    bash run.sh                 # terminal 1: uvicorn on 127.0.0.1:8000
    python web_client.py        # terminal 2: opens http://127.0.0.1:8080

The page guesses `ws://<page-hostname>:8000/turn`, which already matches.


Case 2 — inference server on a remote GPU box, browser on your laptop
---------------------------------------------------------------------
`run.sh` binds uvicorn to `--host 127.0.0.1`, i.e. the server is reachable
only from inside that machine, so forward the port over SSH instead of
exposing it. Run the page locally (not on the server): `getUserMedia` needs a
secure context, and `http://127.0.0.1` is one while `http://<server-ip>` is not.

    # laptop, terminal 1 — keep it open; -N means "no remote command"
    ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
        -L 8000:127.0.0.1:8000 -p <port> -i ~/.ssh/<key> <user>@<server-ip>

    # laptop, terminal 2
    python web_client.py --ws ws://127.0.0.1:8000/turn

Notes:
  * With a `~/.ssh/config` entry the tunnel shortens to
    `ssh -N -L 8000:127.0.0.1:8000 <host-alias>`. The alias must be ASCII —
    OpenSSH rejects a non-ASCII `Host` with "hostname contains invalid
    characters", even though VS Code Remote-SSH accepts it.
  * `bind [::1]:8000: Address already in use` means a tunnel on that port is
    already running; reuse it or pick another local port
    (`-L 8001:127.0.0.1:8000` plus `--ws ws://127.0.0.1:8001/turn`).
  * Sanity check that the tunnel reaches the server:
    `curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/`
    404 is the expected answer — the server only exposes the `/turn`
    WebSocket, so there is no `/` route.
  * If the tunnel drops, the page's WebSocket errors out; restart the tunnel
    and reconnect from the page. This static server needs no restart.

Exposing the inference server directly (`--host 0.0.0.0` in `run.sh`, then
`--ws ws://<server-ip>:8000/turn`) also works — the page stays on
`http://127.0.0.1:8080` for the microphone — but it puts port 8000 on the
network with no auth, so prefer the tunnel.
"""

import argparse
import functools
import http.server
import os
import socketserver
import webbrowser

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class Handler(http.server.SimpleHTTPRequestHandler):
    """Serve `web/` with caching disabled so edits show up on reload."""

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def log_message(self, fmt, *args):
        # Keep the console readable: only report problems.
        if args and str(args[0]).startswith(("GET", "HEAD")) and str(args[1]).startswith("2"):
            return
        super().log_message(fmt, *args)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser(
        description="Serve the SoulX-Duplug web client.",
        epilog=__doc__,  # so `--help` also shows the local / over-SSH recipes
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--ws",
        default=None,
        help="Prefill the inference WebSocket url, e.g. ws://localhost:8000/turn",
    )
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open a browser.")
    args = parser.parse_args()

    if not os.path.isdir(WEB_DIR):
        raise SystemExit(f"web directory not found: {WEB_DIR}")

    handler = functools.partial(Handler, directory=WEB_DIR)

    with Server((args.host, args.port), handler) as httpd:
        shown_host = "127.0.0.1" if args.host in ("0.0.0.0", "") else args.host
        url = f"http://{shown_host}:{args.port}/"
        if args.ws:
            url += f"?ws={args.ws}"

        print(f"[web] serving {WEB_DIR}")
        print(f"[web] open {url}")
        print("[web] make sure the inference server is running (bash run.sh)")
        print("[web] remote server? forward it first: see --help for the SSH tunnel")

        if not args.no_browser:
            webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[web] bye")


if __name__ == "__main__":
    main()
