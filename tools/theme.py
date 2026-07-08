"""Shared visual theme: VSCode Dark+ palette + plotly template + report CSS.

Pure module — no UI framework. Importing it registers and activates the
"vsdark" plotly template. CSS constant is consumed by build_report.py.
"""

import plotly.graph_objects as go
import plotly.io as pio

# VSCode Dark+ editor colors
BG        = "#1e1e1e"   # editor background
BG_PANEL  = "#252526"   # panels / cards
GRID      = "#2d2d2d"   # subtle gridlines / borders
FG        = "#d4d4d4"   # foreground text
FG_DIM    = "#808080"   # secondary text
ACCENT    = "#569cd6"   # keyword blue
GREEN     = "#6a9955"
RED       = "#d16969"
YELLOW    = "#dcdcaa"

# VSCode Dark+ token palette — muted, coherent across all charts/bars
PALETTE = [
    "#569cd6",  # blue (keyword)
    "#4ec9b0",  # teal (type)
    "#ce9178",  # orange (string)
    "#dcdcaa",  # yellow (function)
    "#c586c0",  # purple (control)
    "#9cdcfe",  # light blue (variable)
    "#b5cea8",  # green (number)
    "#d16969",  # red
    "#d7ba7d",  # tan
    "#6a9955",  # green (comment)
]

SANS = '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Helvetica, Arial, sans-serif'
MONO = '"SF Mono", ui-monospace, Menlo, Monaco, "Cascadia Mono", monospace'

_TEMPLATE = go.layout.Template(
    layout=dict(
        font=dict(family=SANS, color=FG, size=12),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        colorway=PALETTE,
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        # Vertical, stacked, anchored inside the top-left. Horizontal legends collapse
        # their entry widths when the webfont paints late (labels overlap — "OptimizedEqual-We");
        # a vertical legend stacks one-per-row with guaranteed spacing, no font-timing race.
        legend=dict(orientation="v", x=0.01, y=0.99, xanchor="left", yanchor="top",
                    bgcolor="rgba(30,30,30,0.85)", bordercolor=GRID, borderwidth=1,
                    font=dict(size=11)),
        hoverlabel=dict(bgcolor="#2d2d30", bordercolor=GRID,
                        font=dict(family=MONO, color=FG, size=12)),
        margin=dict(t=20, b=40, l=50, r=20),
    )
)
pio.templates["vsdark"] = _TEMPLATE
pio.templates.default = "plotly_dark+vsdark"

REPORT_CSS = f"""
:root {{ color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{
    background: {BG}; color: {FG}; margin: 0;
    font-family: {SANS}; font-size: 15px; line-height: 1.55;
}}
main {{ max-width: 1080px; margin: 0 auto; padding: 32px 20px 80px; }}
h1 {{ font-size: 1.6rem; font-weight: 600; letter-spacing: -0.01em; margin: 0 0 4px; }}
h2 {{ font-size: 1.15rem; font-weight: 600; margin: 44px 0 10px; color: {FG};
     border-bottom: 1px solid {GRID}; padding-bottom: 6px; }}
h3 {{ font-size: 0.95rem; font-weight: 600; margin: 18px 0 6px; }}
p, li {{ color: {FG}; }}
.dim {{ color: {FG_DIM}; font-size: 0.85rem; }}
.mono, td.num, th.num {{ font-family: {MONO}; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 10px; margin: 14px 0; }}
.card {{ background: {BG_PANEL}; border: 1px solid {GRID}; border-radius: 6px;
         padding: 10px 14px; }}
.card .k {{ color: {FG_DIM}; font-size: 0.72rem; text-transform: uppercase;
            letter-spacing: 0.05em; }}
.card .v {{ font-family: {MONO}; font-size: 1.25rem; margin-top: 2px; }}
.pos {{ color: {GREEN}; }} .neg {{ color: {RED}; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 0.88rem; }}
th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid {GRID}; }}
th {{ color: {FG_DIM}; font-weight: 500; font-size: 0.78rem; text-transform: uppercase;
     letter-spacing: 0.04em; }}
td.num, th.num {{ text-align: right; white-space: nowrap; }}
.note {{ background: {BG_PANEL}; border-left: 3px solid {ACCENT}; border-radius: 4px;
         padding: 10px 14px; margin: 12px 0; font-size: 0.88rem; }}
.warn {{ border-left-color: {YELLOW}; }}
.chart {{ margin: 8px 0 4px; }}
/* side-by-side comparison (Original | Risk-conscious); collapses on narrow screens */
.cmp {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; align-items: start; margin: 10px 0; }}
.cmp > div {{ min-width: 0; }}          /* let tables shrink instead of overflowing */
.cmp h3 {{ margin-top: 0; }}
.cmp.stretch {{ align-items: stretch; }}                 /* equal-height panels */
.cmp.stretch > .note {{ height: 100%; margin: 0; }}      /* notes fill the cell, same level */
.cmp .pick-sub {{ min-height: 2em; margin-bottom: 6px; }} /* align table tops across columns */
@media (max-width: 760px) {{ .cmp {{ grid-template-columns: 1fr; }} }}
details {{ margin: 10px 0; }}
summary {{ cursor: pointer; color: {ACCENT}; font-size: 0.9rem; }}
a {{ color: {ACCENT}; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
"""
