# ruff: noqa: E501
"""Tema visual, componentes e convenções Plotly do dashboard."""

import plotly.graph_objects as go

INK = "#162C3A"
MUTED = "#62717A"
TEAL = "#188C83"
CORAL = "#D77A5A"
GOLD = "#D3A947"
PAPER = "#F5F6F3"
GRID = "#E3E8E5"


def aplicar_tema(st) -> None:
    st.markdown(
        f"""
        <style>
        :root {{ --ink:{INK}; --muted:{MUTED}; --teal:{TEAL}; --paper:{PAPER}; }}
        html, body, [class*="css"] {{ font-family:'Aptos', 'Segoe UI', sans-serif; color:var(--ink); }}
        .stApp {{ background:var(--paper); }}
        [data-testid="stSidebar"] {{ background:#172D3A; }}
        [data-testid="stSidebar"] * {{ color:#F4F6F4; }}
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{ color:#B7C5C7; }}
        [data-testid="stSidebar"] hr {{ border-color:#35505A; }}
        h1,h2,h3 {{ font-family:'Aptos Display', 'Segoe UI', sans-serif; letter-spacing:-.035em; color:var(--ink); }}
        h1 {{ font-size:2.25rem !important; font-weight:800 !important; }}
        [data-testid="stMetric"] {{ background:#FFFFFF; border:1px solid #E2E8E4; padding:1rem 1.1rem; border-radius:14px; box-shadow:0 3px 14px #19363C0A; }}
        [data-testid="stMetricLabel"] {{ color:var(--muted); font-size:.78rem; }}
        [data-testid="stMetricValue"] {{ font-family:'Manrope',sans-serif; color:var(--ink); }}
        .wg-kicker {{ color:var(--teal); font-size:.72rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }}
        .wg-subtitle {{ color:var(--muted); margin-top:-.7rem; margin-bottom:1.4rem; }}
        .wg-note {{ border-left:3px solid {TEAL}; background:#EAF2EE; padding:.8rem 1rem; border-radius:0 10px 10px 0; color:{INK}; font-size:.9rem; }}
        .wg-brand {{ font-family:'Aptos Display', 'Segoe UI', sans-serif; font-size:1.2rem; font-weight:800; letter-spacing:-.04em; }}
        .wg-brand span {{ color:#61C2B0; }}
        div[data-testid="stPlotlyChart"] {{ background:white; border:1px solid #E2E8E4; border-radius:14px; padding:.45rem; }}
        @media (max-width: 700px) {{ h1 {{ font-size:1.7rem !important; }} [data-testid="stMetric"] {{ padding:.75rem; }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def formatar_figura(fig: go.Figure, altura: int = 440) -> go.Figure:
    """Aplica margens que reservam espaço para título, facetas e legenda externa."""
    fig.update_layout(
        template="plotly_white",
        height=altura,
        margin=dict(l=38, r=28, t=108, b=108),
        font=dict(family="Aptos, Segoe UI, sans-serif", color=INK, size=12),
        paper_bgcolor="white",
        plot_bgcolor="white",
        title=dict(x=0, xanchor="left", y=0.98, yanchor="top", font=dict(size=17)),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.28,
            xanchor="left",
            x=0,
            font=dict(size=11),
            title_text="",
        ),
        hoverlabel=dict(bgcolor=INK, font_color="white"),
    )
    for annotation in fig.layout.annotations or []:
        annotation.update(y=1.09, font=dict(size=12))
    fig.update_xaxes(showgrid=False, linecolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig
