"""Plotly figures used by the app. Colours follow the MarketPulse red palette."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .formatting import change, pct, points
from .tagging import EVENT_TYPES

ACCENT = "#DC2626"
ACCENT_SOFT = "#FCA5A5"
ACCENT_DARK = "#7F1D1D"
GOOD = "#059669"
BAD = "#DC2626"
MUTED = "#94A3B8"
FONT = "Inter, 'Segoe UI', system-ui, sans-serif"
SERIES = ["#DC2626", "#2563EB", "#059669", "#D97706", "#7C3AED", "#0891B2", "#DB2777", "#65A30D", "#475569", "#EA580C"]
EVENT_COLORS = dict(zip(EVENT_TYPES, ["#DC2626", "#059669", "#7C3AED", "#2563EB", "#475569", "#D97706", "#0891B2",
                                      "#DB2777"], strict=True))
REDS = [[0, "#FFF5F5"], [0.15, "#FECACA"], [0.5, "#F87171"], [1, "#991B1B"]]


def colors_for(names: list[str]) -> dict[str, str]:
    return {name: SERIES[i % len(SERIES)] for i, name in enumerate(names)}


def short(text: str, n: int = 38) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip(" ·") + "…"


def _style(fig: go.Figure, height: int = 340, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        title=dict(text=title, x=0, font=dict(size=15)) if title else None,
        margin=dict(l=8, r=8, t=48 if title else 16, b=8),
        font=dict(family=FONT, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_family=FONT),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.25)", zeroline=False)
    return fig


def _legend_below(fig: go.Figure) -> go.Figure:
    """Many-series charts keep their legend under the plot so it never collides with the title."""
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="left", x=0, font=dict(size=11)),
                      margin=dict(b=70))
    return fig


# ------------------------------------------------------------------------ competitors
def sov_bar(sov: pd.DataFrame, colors: dict[str, str], days: int) -> go.Figure:
    data = sov.iloc[::-1]
    fig = go.Figure(
        go.Bar(
            y=data["competitor"],
            x=data["share"],
            orientation="h",
            marker_color=[colors[c] for c in data["competitor"]],
            text=[f"{pct(s)} · {m:,}" for s, m in zip(data["share"], data["mentions"], strict=True)],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.1%} of competitor mentions<extra></extra>",
        )
    )
    fig.update_xaxes(tickformat=".0%", range=[0, max(0.05, float(sov["share"].max()) * 1.3)], showgrid=True,
                     gridcolor="rgba(148,163,184,0.25)")
    return _style(fig, height=300, title=f"Share of voice, last {days} days")


def momentum_bar(mom: pd.DataFrame, days: int) -> go.Figure:
    data = mom.iloc[::-1]
    fig = go.Figure(
        go.Bar(
            y=data["competitor"],
            x=data["sov_change"],
            orientation="h",
            marker_color=[GOOD if v >= 0 else BAD for v in data["sov_change"]],
            text=[f"{points(v)} · {lst} vs {pr}" for v, lst, pr in
                  zip(data["sov_change"], data["last"], data["prior"], strict=True)],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{text}<extra></extra>",
        )
    )
    span = max(0.02, float(data["sov_change"].abs().max()) * 2.6)
    fig.update_xaxes(tickformat="+.0%", range=[-span, span], zeroline=True, zerolinecolor=MUTED)
    return _style(fig, height=300, title=f"Share-of-voice change: last {days} days vs the {days} before")


def event_heatmap(events: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=events.values,
            x=list(events.columns),
            y=list(events.index),
            colorscale=REDS,
            showscale=False,
            text=events.values,
            texttemplate="%{text}",
            hovertemplate="%{y} · %{x}: %{z} headlines<extra></extra>",
            xgap=3,
            ygap=3,
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(side="top")
    return _style(fig, height=70 + 46 * len(events), title=None)


def sentiment_bars(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for label, color in (("negative", BAD), ("neutral", "#CBD5E1"), ("positive", GOOD)):
        fig.add_bar(y=table["competitor"], x=table[label], name=label.title(), orientation="h", marker_color=color)
    fig.update_layout(barmode="stack")
    fig.update_yaxes(autorange="reversed")
    return _style(fig, height=280, title="Model-judged tone of evidence headlines")


# ----------------------------------------------------------------------------- trends
def daily_lines(daily: pd.DataFrame, names: list[str], colors: dict[str, str], spikes: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for name in names:
        fig.add_scatter(
            x=daily.index, y=daily[name], mode="lines", name=name, line=dict(color=colors[name], width=2.2),
            hovertemplate=f"{name} · %{{x|%d %b}}: %{{y}} headlines<extra></extra>",
        )
    if len(spikes):
        fig.add_scatter(
            x=pd.to_datetime(spikes["date"]), y=spikes["count"], mode="markers", name="Spike",
            marker=dict(size=13, color="rgba(0,0,0,0)", line=dict(color=ACCENT_DARK, width=2.5)),
            customdata=spikes[["competitor", "z"]].values,
            hovertemplate="%{customdata[0]} spike · %{x|%d %b}: %{y} headlines "
                          "(z = %{customdata[1]:.1f})<extra></extra>",
        )
    fig.update_yaxes(title="Headlines per day", rangemode="tozero")
    return _style(fig, height=380, title="Daily headlines per competitor (spikes circled)")


def weekly_share_area(weekly: pd.DataFrame, names: list[str], colors: dict[str, str]) -> go.Figure:
    fig = go.Figure()
    for name in names:
        fig.add_scatter(
            x=weekly["week_end"], y=weekly[name], mode="lines", name=name, stackgroup="sov",
            line=dict(width=0.5, color=colors[name]), fillcolor=colors[name],
            hovertemplate=f"{name} · week to %{{x|%d %b}}: %{{y:.0%}}<extra></extra>",
        )
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig = _style(fig, height=360, title="Weekly share of voice (7-day blocks)")
    return _legend_below(fig)


def event_volume(articles: pd.DataFrame, start, end) -> go.Figure:
    days = pd.date_range(start, end, freq="D")
    fig = go.Figure()
    exploded = articles[["day", "events"]].explode("events").dropna(subset=["events"])
    exploded["day"] = pd.to_datetime(exploded["day"])
    for event in EVENT_TYPES:
        counts = exploded.loc[exploded["events"] == event, "day"].value_counts().reindex(days, fill_value=0)
        fig.add_bar(x=days, y=counts.values, name=event, marker_color=EVENT_COLORS[event],
                    hovertemplate=f"{event} · %{{x|%d %b}}: %{{y}}<extra></extra>")
    fig.update_layout(barmode="stack", bargap=0.15)
    fig.update_yaxes(title="Tagged headlines")
    fig = _style(fig, height=360, title="Rule-tagged events per day")
    return _legend_below(fig)


# ----------------------------------------------------------------------------- themes
def theme_bars(themes: pd.DataFrame, names: dict[int, str]) -> go.Figure:
    data = themes.iloc[::-1]
    labels = [short(f"{int(t) + 1}. {names.get(int(t), lab)}", 44) for t, lab in
              zip(data["theme"], data["label"], strict=True)]
    colors = [MUTED if c is not None and c <= 0 else (ACCENT if c is None or c >= 0.5 else ACCENT_SOFT)
              for c in data["change"]]
    fig = go.Figure(
        go.Bar(
            y=labels, x=data["size"], orientation="h", marker_color=colors,
            text=[f"{s:,} · {change(c)}" for s, c in zip(data["size"], data["change"], strict=True)],
            textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x} headlines<extra></extra>",
        )
    )
    fig.update_xaxes(range=[0, float(themes["size"].max()) * 1.3], showgrid=True, gridcolor="rgba(148,163,184,0.25)")
    return _style(fig, height=110 + 34 * len(themes), title="Themes by size (week-on-week change)")


def silhouette_chart(sil: pd.DataFrame, k: int, auto: bool) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=sil["k"], y=sil["score"], mode="lines+markers", line=dict(color=MUTED, width=2),
                    marker=dict(size=7, color=MUTED), name="Silhouette",
                    hovertemplate="k = %{x}: %{y:.3f}<extra></extra>")
    chosen = sil[sil["k"] == k]
    if len(chosen):
        fig.add_scatter(x=chosen["k"], y=chosen["score"], mode="markers", marker=dict(size=15, color=ACCENT),
                        name="Chosen k", hovertemplate="chosen k = %{x}: %{y:.3f}<extra></extra>")
    fig.update_xaxes(title="Number of themes (k)", dtick=1)
    fig.update_yaxes(title="Cosine silhouette")
    fig.update_layout(showlegend=False)
    return _style(fig, height=300, title="How k was chosen" if auto else "Silhouette by k (k fixed in settings)")


def theme_competitor_heatmap(articles: pd.DataFrame, themes: pd.DataFrame, names: list[str],
                             theme_names: dict[int, str]) -> go.Figure:
    rows = []
    for t in themes["theme"]:
        sub = articles[articles["theme"] == t]
        rows.append([int(sub["competitors"].map(lambda cs, n=n: n in cs).sum()) for n in names])
    labels = [short(f"{int(t) + 1}. {theme_names.get(int(t), lab)}", 40) for t, lab in
              zip(themes["theme"], themes["label"], strict=True)]
    fig = go.Figure(
        go.Heatmap(z=rows, x=names, y=labels, colorscale=REDS, showscale=False, text=rows, texttemplate="%{text}",
                   xgap=3, ygap=3, hovertemplate="%{y} · %{x}: %{z} headlines<extra></extra>")
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(side="top")
    return _style(fig, height=90 + 36 * len(labels), title=None)


# ---------------------------------------------------------------------------- sources
def outlets_bar(outlets: pd.DataFrame, n: int = 12) -> go.Figure:
    data = outlets.head(n).iloc[::-1]
    fig = go.Figure(
        go.Bar(y=[short(s, 30) for s in data["source"]], x=data["headlines"], orientation="h",
               marker_color=ACCENT_SOFT, text=data["headlines"], textposition="outside", cliponaxis=False,
               hovertemplate="%{y}: %{x} headlines<extra></extra>")
    )
    fig.update_xaxes(range=[0, float(data["headlines"].max()) * 1.2 if len(data) else 1])
    return _style(fig, height=80 + 26 * len(data), title=f"Top {min(n, len(outlets))} outlets")
