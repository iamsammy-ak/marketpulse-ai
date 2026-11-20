"""Small HTML/CSS helpers for a polished Streamlit look (works in light and dark themes)."""

from __future__ import annotations

import html

import streamlit as st

ACCENT = "#DC2626"
ACCENT_DARK = "#7F1D1D"

_CSS = f"""
<style>
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1240px; }}
.mp-hero {{
  border-radius: 22px; padding: 30px 34px; margin-bottom: 1.2rem; color: #fff;
  background:
    radial-gradient(circle at 88% 18%, rgba(255,255,255,0.20), transparent 38%),
    radial-gradient(circle at 10% 110%, rgba(255,255,255,0.10), transparent 40%),
    linear-gradient(135deg, {ACCENT} 0%, {ACCENT_DARK} 100%);
  box-shadow: 0 18px 40px -22px rgba(127,29,29,0.65);
}}
.mp-hero .mp-title {{ font-size: 2.15rem; font-weight: 750; letter-spacing: -0.02em; line-height: 1.15; }}
.mp-hero .mp-sub {{ margin-top: 8px; font-size: 1.03rem; opacity: 0.92; max-width: 800px; }}
.mp-chip {{
  display: inline-block; margin: 14px 8px 0 0; padding: 4px 12px; border-radius: 999px;
  background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.22); font-size: 0.8rem;
}}
div[data-testid="stMetric"] {{
  border: 1px solid rgba(220,38,38,0.20); border-radius: 16px; padding: 14px 16px;
  background: linear-gradient(180deg, rgba(220,38,38,0.06), rgba(220,38,38,0.015));
}}
div[data-testid="stMetricLabel"] p {{ font-size: 0.82rem; opacity: 0.8; }}
div[data-testid="stMetricValue"] {{ font-size: 1.6rem; font-weight: 650; }}
.mp-card {{
  border: 1px solid rgba(148,163,184,0.35); border-radius: 16px; padding: 14px 18px; height: 100%;
  margin-bottom: 14px;
}}
.mp-card h4 {{ margin: 0 0 4px 0; font-size: 1.0rem; }}
.mp-card .mp-meta {{ font-size: 0.82rem; opacity: 0.7; margin-bottom: 8px; }}
.mp-card ul {{ margin: 6px 0 0 0; padding-left: 1.1rem; }}
.mp-card li {{ margin-bottom: 3px; }}
.mp-card p {{ margin: 6px 0; }}
.mp-pill {{
  display: inline-block; padding: 5px 14px; border-radius: 999px; font-weight: 600; font-size: 0.9rem;
  background: rgba(220,38,38,0.10); color: {ACCENT_DARK}; border: 1px solid rgba(220,38,38,0.25);
}}
.mp-pill-good {{ background: rgba(5,150,105,0.10); color: #065F46; border-color: rgba(5,150,105,0.3); }}
.mp-table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; margin: 0.2rem 0 1rem 0; }}
.mp-table th {{
  text-align: left; font-weight: 600; opacity: 0.7; padding: 8px 10px;
  border-bottom: 1px solid rgba(148,163,184,0.45);
}}
.mp-table td {{ padding: 9px 10px; border-bottom: 1px solid rgba(148,163,184,0.2); vertical-align: top; }}
.mp-table tr:hover td {{ background: rgba(220,38,38,0.03); }}
.mp-table a {{ color: inherit; text-decoration: none; border-bottom: 1px solid rgba(220,38,38,0.35); }}
.mp-table a:hover {{ color: {ACCENT}; }}
.mp-tag {{
  display: inline-block; padding: 1px 8px; margin: 1px 4px 2px 0; border-radius: 999px; font-size: 0.76rem;
  white-space: nowrap; border: 1px solid rgba(148,163,184,0.45);
}}
.mp-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 99px; margin-right: 5px; }}
.mp-bar {{
  display: inline-block; width: 64px; height: 8px; border-radius: 99px; overflow: hidden;
  background: rgba(220,38,38,0.13); vertical-align: middle; margin-right: 8px;
}}
.mp-bar > div {{ height: 100%; border-radius: 99px; background: {ACCENT}; }}
.mp-num {{ font-weight: 650; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.mp-muted {{ opacity: 0.6; font-size: 0.82rem; }}
.mp-up {{ color: #059669; font-weight: 650; white-space: nowrap; }}
.mp-down {{ color: #DC2626; font-weight: 650; white-space: nowrap; }}
.mp-ref {{ font-weight: 650; color: {ACCENT}; white-space: nowrap; }}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, chip_list: list[str]) -> None:
    chip_html = "".join(f'<span class="mp-chip">{html.escape(c)}</span>' for c in chip_list)
    st.markdown(
        f'<div class="mp-hero"><div class="mp-title">{html.escape(title)}</div>'
        f'<div class="mp-sub">{html.escape(subtitle)}</div>{chip_html}</div>',
        unsafe_allow_html=True,
    )


def pill(text: str, good: bool = False) -> None:
    cls = "mp-pill mp-pill-good" if good else "mp-pill"
    st.markdown(f'<span class="{cls}">{html.escape(text)}</span>', unsafe_allow_html=True)


def table(headers: list[str], rows: list[list], html_cols: tuple[int, ...] = ()) -> None:
    """Render a wrapping HTML table. Cells are escaped unless their column is in `html_cols`."""
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{cell if i in html_cols else html.escape(str(cell))}</td>" for i, cell in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    st.markdown(f'<table class="mp-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>',
                unsafe_allow_html=True)


def link(text: str, url: str) -> str:
    return f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener">{html.escape(text)}</a>'


def tags(items: list[str], colors: dict[str, str] | None = None) -> str:
    """Small rounded labels, optionally with a coloured dot per item."""
    out = []
    for item in items:
        dot = f'<span class="mp-dot" style="background:{colors[item]}"></span>' if colors and item in colors else ""
        out.append(f'<span class="mp-tag">{dot}{html.escape(item)}</span>')
    return "".join(out) or '<span class="mp-muted">—</span>'


def share_bar(value: float) -> str:
    width = max(0.0, min(100.0, value * 100))
    return (f'<span style="white-space:nowrap"><span class="mp-bar"><div style="width:{width:.0f}%"></div></span>'
            f'<span class="mp-num">{value * 100:.0f}%</span></span>')


def signed(text: str, value: float | None, tolerance: float = 5e-4) -> str:
    """Green for gains, red for losses, neutral when the value rounds to zero."""
    if value is None or value > tolerance:
        cls = "mp-up"
    elif value < -tolerance:
        cls = "mp-down"
    else:
        cls = "mp-num"
    return f'<span class="{cls}">{html.escape(text)}</span>'


def nowrap(text: str) -> str:
    return f'<span style="white-space:nowrap">{html.escape(text)}</span>'


def muted(text: str) -> str:
    return f'<span class="mp-muted">{html.escape(text)}</span>'


def ref(n: int) -> str:
    return f'<span class="mp-ref">[{n}]</span>' if n else ""


def card(title: str, meta: str, body_html: str) -> None:
    st.markdown(
        f'<div class="mp-card"><h4>{html.escape(title)}</h4><div class="mp-meta">{meta}</div>{body_html}</div>',
        unsafe_allow_html=True,
    )
