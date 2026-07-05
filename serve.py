"""Local live server for the dashboard — the primary local experience.

Serves both pages (portfolio monitor + Pairs Trading Lab). A normal page load is
instant: it builds from the on-disk data buffer (tools/data_buffer.py) with no
network. The "↻ Update to now" button forces a live re-fetch; a failed fetch
keeps the last-good buffered values and flags staleness (never the cost-basis
garbage a silent fetch failure used to show).

  .venv/bin/python serve.py        # then open http://localhost:8000

The static build (build_report.py / build_pairs_report.py → docs/) is unchanged
and exists only to publish the GitHub Pages snapshot.
"""

import http.server
import socketserver
import webbrowser
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import build_report as B
import build_pairs_report as P
import build_scenarios_report as S
import build_strategy_report as ST   # the Strategy page folds the momentum lab in as its lower half
import build_vol_report as V         # vol lab — forecast-based volatility targeting
import build_edge_report as E        # edge stack — structural edges (capacity/flows/horizon/costs)
import build_econo_report as EC      # econophysics lab — signals vs the production gate

PORT = 8000

# route → loader path, live build path, last-good snapshot file
PAGES = {
    "main":     dict(loader="/",         build="/report",          snap=B.ROOT / "local/report.html"),
    "pairs":    dict(loader="/pairs",    build="/pairs-report",    snap=P.ROOT / "local/pairs.html"),
    "strategy": dict(loader="/strategy", build="/strategy-report", snap=ST.ROOT / "local/strategy.html"),
    "scenarios": dict(loader="/scenarios", build="/scenarios-report", snap=S.ROOT / "local/scenarios.html"),
    "vol":      dict(loader="/vol",      build="/vol-report",       snap=V.ROOT / "local/vol.html"),
    "edge":     dict(loader="/edge",     build="/edge-report",      snap=E.ROOT / "local/edge.html"),
    "econo":    dict(loader="/econo",    build="/econo-report",     snap=EC.OUT),
}

# cross-page nav links shown in the top bar, in display order.
# Pairs / Momentum-lab / Scenarios routes still exist (PAGES below) but are
# unlinked — the live site is just Portfolio + the chosen Strategy.
_NAV = [
    ("main",     "/",         "Portfolio"),
    ("strategy", "/strategy", "Strategy"),
    ("vol",      "/vol",      "Vol"),
    ("edge",     "/edge",     "Edge"),
    ("econo",    "/econo",    "Econo"),
]


def _loader(build_path: str) -> str:
    """Spinner page; the report loads in a hidden iframe (so Plotly's inline
    scripts run), then we reveal it."""
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Updating…</title>
<style>
body{{background:#1e1e1e;color:#d4d4d4;margin:0;height:100vh;display:flex;flex-direction:column;
 align-items:center;justify-content:center;font-family:-apple-system,"Segoe UI",sans-serif}}
.sp{{width:40px;height:40px;border:3px solid #2d2d2d;border-top-color:#569cd6;border-radius:50%;
 animation:s .8s linear infinite}}@keyframes s{{to{{transform:rotate(360deg)}}}}
p{{color:#808080;font-size:14px;margin-top:16px}}
iframe{{position:fixed;inset:0;width:100%;height:100%;border:0;background:#1e1e1e;
 opacity:0;pointer-events:none}}
</style></head><body>
<iframe id="f" src="{build_path}"></iframe>
<div class="sp" id="spin"></div><p id="msg">Loading…</p>
<script>
// Render the iframe at full viewport width (opacity:0, NOT display:none) so Plotly
// measures real widths while building — a display:none iframe has zero width and clips
// every legend box to one char. Reveal + nudge a resize once the report has loaded.
var f=document.getElementById('f');
f.onload=function(){{document.getElementById('spin').style.display='none';
 document.getElementById('msg').style.display='none';
 f.style.opacity='1';f.style.pointerEvents='auto';
 try{{f.contentWindow.dispatchEvent(new Event('resize'));}}catch(e){{}}}};
f.onerror=function(){{document.getElementById('msg').textContent='Build failed — see terminal.';}};
</script></body></html>"""


def _bar(page: str, as_of: str | None = None, stale: dict | None = None,
         stale_msg: str = "", note: str = "") -> str:
    """Fixed top bar injected into a served report: status + nav + force-refresh."""
    force_url = PAGES[page]["loader"] + "?force=1"
    link = "color:#569cd6;text-decoration:none;font-weight:600"
    nav_links = " · ".join(
        f'<a href="{href}" target="_top" style="{link}">{label}</a>' if name != page
        else f'<span style="color:#d4d4d4">{label}</span>'
        for name, href, label in _NAV
    )

    if stale_msg:                                          # genuine failure (warning)
        status = f'<span style="color:#d7ba7d">⚠ {stale_msg}</span>'
    elif stale:                                            # some live prices fell back
        names = ", ".join(sorted(stale))
        status = (f'<span style="color:#d7ba7d">⚠ {len(stale)} price(s) stale '
                  f'({names}) — ↻ to retry</span>')
    elif note:                                             # neutral (snapshot load)
        status = f'<span>{note}</span>'
    elif as_of:
        status = f'<span>live · {as_of[11:19]}</span>'
    else:
        status = f'<span>live · {datetime.now():%H:%M:%S}</span>'

    return f"""<div style="position:fixed;top:0;right:0;z-index:99999;background:#252526;
border:1px solid #2d2d2d;border-radius:0 0 0 6px;padding:6px 12px;display:flex;gap:14px;
align-items:center;font-family:'SF Mono',ui-monospace,Menlo,monospace;font-size:12px;color:#808080">
{status}
{nav_links}
<a href="{force_url}" target="_top" style="{link}">↻ Update to now</a>
</div>"""


def _inject(html: str, page: str, as_of: str | None = None,
            stale: dict | None = None, stale_msg: str = "", note: str = "") -> str:
    # Rewrite the static cross-links the build emits into server routes that
    # break out of the loader iframe.
    html = (html.replace("href='pairs.html'", "href='/pairs' target='_top'")
                .replace("href='momentum.html'", "href='/momentum' target='_top'")
                .replace("href='report.html'", "href='/' target='_top'")
                .replace("href='index.html'", "href='/' target='_top'"))
    return html.replace("<main>", "<main>" + _bar(page, as_of, stale, stale_msg, note), 1)


class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, html: str):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_report(self, page: str, force: bool):
        cfg = PAGES[page]
        # Normal load: serve the last rendered snapshot instantly (no rebuild, no
        # recompute). Only a forced refresh (or a cold/missing snapshot) rebuilds.
        if not force and cfg["snap"].exists():
            when = datetime.fromtimestamp(cfg["snap"].stat().st_mtime).strftime("%H:%M")
            self._send(_inject(cfg["snap"].read_text(), page,
                               note=f"snapshot {when} · ↻ for live"))
            return
        src = "live prices" if force else "first build"
        print(f"[{datetime.now():%H:%M:%S}] building {page} ({src})…")
        try:
            if page == "main":
                d = B.gather(force=force)
                html = B.build(d, public=False)
                stale, as_of = d.get("stale"), d.get("as_of")
            elif page == "pairs":
                d = P.gather(force=force)
                html = P.build(d, public=False)
                stale, as_of = None, None
            elif page == "strategy":
                d = ST.gather(force=force)
                html = ST.build(d, public=False)
                stale, as_of = None, None
            elif page == "vol":
                d = V.gather(force=force)
                html = V.build(d, public=False)
                stale, as_of = None, None
            elif page == "edge":
                d = E.gather(force=force)
                html = E.build(d, public=False)
                stale, as_of = None, None
            elif page == "econo":
                d = EC.gather()          # several minutes: full experiment re-run
                html = EC.build(d, public=False)
                stale, as_of = None, None
            else:  # scenarios (local-only)
                d = S.gather(force=force)
                html = S.build(d, public=False)
                stale, as_of = None, None
            cfg["snap"].write_text(html)            # keep last-good snapshot fresh
            html = _inject(html, page, as_of=as_of, stale=stale)
            print("  ok")
        except Exception as e:
            print(f"  build failed ({e}); serving last snapshot")
            if cfg["snap"].exists():
                when = datetime.fromtimestamp(cfg["snap"].stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                html = _inject(cfg["snap"].read_text(), page,
                               stale_msg=f"build failed — snapshot {when}")
            else:
                html = ("<body style='background:#1e1e1e;color:#d16969;font-family:monospace;"
                        f"padding:40px'>build failed and no snapshot exists yet:<br>{e}</body>")
        self._send(html)

    def do_GET(self):
        u = urlparse(self.path)
        force = parse_qs(u.query).get("force", ["0"])[0] in ("1", "true")
        path = u.path

        if path in ("/", "/index.html"):
            self._send(_loader("/report" + ("?force=1" if force else "")))
        elif path == "/pairs":
            self._send(_loader("/pairs-report" + ("?force=1" if force else "")))
        elif path == "/strategy":
            self._send(_loader("/strategy-report" + ("?force=1" if force else "")))
        elif path == "/scenarios":
            self._send(_loader("/scenarios-report" + ("?force=1" if force else "")))
        elif path == "/vol":
            self._send(_loader("/vol-report" + ("?force=1" if force else "")))
        elif path == "/edge":
            self._send(_loader("/edge-report" + ("?force=1" if force else "")))
        elif path == "/econo":
            self._send(_loader("/econo-report" + ("?force=1" if force else "")))
        elif path == "/report":
            self._serve_report("main", force)
        elif path == "/pairs-report":
            self._serve_report("pairs", force)
        elif path == "/strategy-report":
            self._serve_report("strategy", force)
        elif path == "/scenarios-report":
            self._serve_report("scenarios", force)
        elif path == "/vol-report":
            self._serve_report("vol", force)
        elif path == "/edge-report":
            self._serve_report("edge", force)
        elif path == "/econo-report":
            self._serve_report("econo", force)
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass  # quiet; we print our own line on rebuild


def main():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}"
        print(f"Live dashboard at {url}  (refresh = instant from buffer; "
              f"↻ Update to now = live fetch; Ctrl-C to stop)")
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
