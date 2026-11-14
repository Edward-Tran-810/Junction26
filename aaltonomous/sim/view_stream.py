"""Read-only browser viewer for a running Aalto sim's head RGB-D stream (no WebRTC needed).

    python -m aaltonomous.sim.view_stream --rgbd-port 56560 --http-port 8765        # instance b
    python -m aaltonomous.sim.view_stream --rgbd-port 55560 --http-port 8765        # instance a

It only SUBSCRIBES to the sim's head RGB-D publisher (interface/head_rgbd.HeadRGBDClient, zmq SUB): it never
resets, randomises or commands the sim, so it is safe next to someone else's runs. It serves an MJPEG page on
127.0.0.1 only (not on the network): reach it from another PC through an SSH tunnel:

    ssh -N -L 8765:127.0.0.1:8765 <user>@<server>          # then open http://localhost:8765

Optional --perception draws the current perception (config.yaml) detections on the colour image.
"""
from __future__ import annotations

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

PAGE = b"""<!doctype html><html><head><title>sim head camera</title>
<style>body{background:#111;color:#ddd;font-family:sans-serif;margin:16px}img{max-width:100%;height:auto}</style>
</head><body><div>Aalto sim head camera (left: RGB, right: depth). Read-only.</div>
<img src="/stream.mjpg"></body></html>"""


def main(argv=None):
    from aaltonomous.interface.head_rgbd import HeadRGBDClient
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="sim host")
    ap.add_argument("--rgbd-port", type=int, default=56560, help="head RGB-D port (instance a 55560, b 56560)")
    ap.add_argument("--http-port", type=int, default=8765)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--perception", action="store_true", help="draw perception detections (config.yaml)")
    a = ap.parse_args(argv)
    cli = HeadRGBDClient(a.host, a.rgbd_port)
    per = None
    if a.perception:
        from aaltonomous.baselines.classical.loop.vision.eval_vision import make_perception
        per = make_perception("v2")
    state = {"jpg": None}

    def render():
        while True:
            got = cli.get()
            if got is None:
                img = np.zeros((480, 1280, 3), np.uint8)
                cv2.putText(img, f"waiting for frames on :{a.rgbd_port} ...", (20, 240), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (255, 255, 255), 2)
            else:
                rgb, depth, stamp, fid = got
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if per is not None:
                    per.reset_calibration()
                    for d in per.detect(rgb, depth):
                        u, v = int(d.uv[0]), int(d.uv[1])
                        cv2.circle(bgr, (u, v), 8, (0, 255, 0), 2)
                        cv2.putText(bgr, f"{d.color or '?'} {d.shape or '?'}", (u + 10, v - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                dv = np.clip((depth.astype(np.float32) - 300) / (1500 - 300), 0, 1)
                dcol = cv2.applyColorMap((255 * (1 - dv)).astype(np.uint8), cv2.COLORMAP_TURBO)
                dcol[depth == 0] = 0
                img = np.hstack([bgr, cv2.resize(dcol, (bgr.shape[1], bgr.shape[0]))])
                cv2.putText(img, f"frame {fid}  age {cli.age():.2f}s", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)
            state["jpg"] = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            time.sleep(1.0 / a.fps)

    threading.Thread(target=render, daemon=True).start()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path.startswith("/stream.mjpg"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        j = state["jpg"]
                        if j:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n")
                        time.sleep(1.0 / a.fps)
                except (BrokenPipeError, ConnectionResetError):
                    return
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(PAGE)

    srv = ThreadingHTTPServer(("127.0.0.1", a.http_port), H)
    print(f"viewing head RGB-D :{a.rgbd_port} -> http://127.0.0.1:{a.http_port}  (tunnel this port; Ctrl+C stops)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
