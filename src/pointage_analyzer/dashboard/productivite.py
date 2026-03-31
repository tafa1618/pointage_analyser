"""
Onglet Productivité — dashboard Streamlit.

Consomme un ProductiviteResult produit par ProductiviteBuilder.
Aucun calcul métier ici — uniquement affichage.

Structure de l'onglet :
  1. KPI cards globales YTD
  2. Évolution mensuelle (courbe)
  3. Classement par équipe (barres horizontales)
  4. Matrice technicien × mois (heatmap)
  5. Tableau détaillé techniciens (filtrable)
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

# Import relatif — ajuster si le module est dans un sous-package différent
try:
    from pointage_analyzer.pipeline.productivite_builder import (
        ProductiviteResult,
        SEUIL_EXCELLENT,
        SEUIL_BON,
        SEUIL_FAIBLE,
    )
except ImportError:
    # Fallback pour exécution standalone / tests
    from productivite_builder import (  # type: ignore[no-redef]
        ProductiviteResult,
        SEUIL_EXCELLENT,
        SEUIL_BON,
        SEUIL_FAIBLE,
    )

# ─── Couleurs & constantes ────────────────────────────────────────────────
_COULEUR_EXCELLENT = "#28a745"
_COULEUR_BON       = "#fd7e14"
_COULEUR_FAIBLE    = "#dc3545"
_COULEUR_CRITIQUE  = "#6c757d"
_COULEUR_PRIMAIRE  = "#002060"   # Navy CAT
_COULEUR_ACCENT    = "#FFCD11"   # Jaune CAT

_PALETTE_EQUIPES = [
    "#002060", "#FFCD11", "#1f77b4", "#ff7f0e", "#2ca02c",
    "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#aec7e8",
]


def _pct(val: float) -> str:
    return f"{val:.1%}"


def _couleur_perf(ratio: float) -> str:
    if ratio >= SEUIL_EXCELLENT:
        return _COULEUR_EXCELLENT
    elif ratio >= SEUIL_BON:
        return _COULEUR_BON
    elif ratio >= SEUIL_FAIBLE:
        return _COULEUR_FAIBLE
    return _COULEUR_CRITIQUE


def render_productivite_tab(result: ProductiviteResult) -> None:
    """
    Point d'entrée appelé depuis app.py.

    Usage dans app.py :
        from pointage_analyzer.dashboard.productivite import render_productivite_tab
        with tab_productivite:
            render_productivite_tab(pipeline_result.productivite)
    """
    if result is None or result.ytd_hr_totale == 0:
        st.info("Aucune donnée de productivité disponible. Lancez l'analyse d'abord.")
        return

    # ── En-tête ──────────────────────────────────────────────────────────
    st.markdown("## ⚡ Productivité Techniciens")
    st.caption(
        f"Période : **{result.periode_debut}** → **{result.periode_fin}** | "
        f"{result.nb_techniciens} techniciens | {result.nb_equipes} équipes"
    )

    # ── Section 1 : KPI cards YTD ─────────────────────────────────────────
    _render_kpi_cards(result)

    st.divider()

    # ── Section 2 : Évolution mensuelle ──────────────────────────────────
    _render_evolution_mensuelle(result)

    st.divider()

    # ── Section 3 : Productivité par équipe ──────────────────────────────
    col_eq, col_eq_mois = st.columns([1, 1])
    with col_eq:
        _render_barres_equipes(result)
    with col_eq_mois:
        _render_equipe_mois(result)

    st.divider()

    # ── Section 4 : Matrice technicien × mois ────────────────────────────
    _render_matrice_heatmap(result)

    st.divider()

    # ── Section 5 : Tableau détaillé techniciens ─────────────────────────
    _render_tableau_techniciens(result)


# ─────────────────────────────────────────────────────────────────────────────
# Composants internes
# ─────────────────────────────────────────────────────────────────────────────

def _render_kpi_cards(result: ProductiviteResult) -> None:
    """4 métriques globales YTD."""
    prod = result.ytd_productivite
    couleur = _couleur_perf(prod)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            label="🎯 Productivité YTD",
            value=_pct(prod),
            help="Σ Heures Facturables / Σ Hr_Totale",
        )
        # Barre de progression colorée
        st.markdown(
            f"""
            <div style="background:#e9ecef;border-radius:4px;height:8px;margin-top:-12px">
              <div style="background:{couleur};width:{min(prod,1)*100:.0f}%;
                          height:8px;border-radius:4px"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.metric(
            label="⏱ Heures Facturables",
            value=f"{result.ytd_facturable:,.0f}h",
        )

    with c3:
        st.metric(
            label="🕐 Heures Totales",
            value=f"{result.ytd_hr_totale:,.0f}h",
        )

    with c4:
        heures_non_fact = result.ytd_hr_totale - result.ytd_facturable
        st.metric(
            label="📉 Heures Non Fact.",
            value=f"{heures_non_fact:,.0f}h",
            help="Hr_Totale − Facturable",
        )


def _render_evolution_mensuelle(result: ProductiviteResult) -> None:
    """Courbe d'évolution mensuelle avec Chart.js via st.components."""
    st.markdown("### 📈 Évolution mensuelle")

    df = result.par_mois
    if df.empty:
        st.info("Pas de données mensuelles.")
        return

    mois_labels = df["mois"].tolist()
    prod_values = [round(v * 100, 1) for v in df["productivite"].tolist()]
    fact_values = [round(v, 1) for v in df["facturable"].tolist()]
    tot_values  = [round(v, 1) for v in df["hr_totale"].tolist()]

    html = f"""
    <div style="position:relative;height:320px">
      <canvas id="chartMois"></canvas>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script>
    const ctx = document.getElementById('chartMois').getContext('2d');
    new Chart(ctx, {{
      data: {{
        labels: {mois_labels},
        datasets: [
          {{
            type: 'bar',
            label: 'Heures Facturables',
            data: {fact_values},
            backgroundColor: 'rgba(255, 205, 17, 0.7)',
            yAxisID: 'yH',
            order: 2,
          }},
          {{
            type: 'bar',
            label: 'Heures Totales',
            data: {tot_values},
            backgroundColor: 'rgba(0, 32, 96, 0.2)',
            yAxisID: 'yH',
            order: 3,
          }},
          {{
            type: 'line',
            label: 'Productivité (%)',
            data: {prod_values},
            borderColor: '#dc3545',
            backgroundColor: 'rgba(220,53,69,0.1)',
            borderWidth: 3,
            pointRadius: 6,
            pointBackgroundColor: '#dc3545',
            tension: 0.3,
            yAxisID: 'yP',
            order: 1,
          }}
        ]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{ position: 'top' }},
          tooltip: {{
            callbacks: {{
              label: function(ctx) {{
                if (ctx.dataset.label === 'Productivité (%)') return ctx.dataset.label + ': ' + ctx.raw + '%';
                return ctx.dataset.label + ': ' + ctx.raw + 'h';
              }}
            }}
          }}
        }},
        scales: {{
          yH: {{ type: 'linear', position: 'left',  title: {{ display: true, text: 'Heures' }} }},
          yP: {{ type: 'linear', position: 'right', title: {{ display: true, text: 'Productivité (%)' }},
                 min: 0, max: 100,
                 grid: {{ drawOnChartArea: false }} }}
        }}
      }}
    }});
    </script>
    """
    st.components.v1.html(html, height=340)


def _render_barres_equipes(result: ProductiviteResult) -> None:
    """Barres horizontales productivité par équipe YTD."""
    st.markdown("### 🏭 Par équipe (YTD)")

    df = result.par_equipe.sort_values("productivite", ascending=True)
    if df.empty:
        st.info("Pas de données équipes.")
        return

    equipes   = df["equipe"].tolist()
    prod_vals = [round(v * 100, 1) for v in df["productivite"].tolist()]
    colors    = [_couleur_perf(v / 100) for v in prod_vals]

    html = f"""
    <div style="position:relative;height:{max(250, len(equipes)*40)}px">
      <canvas id="chartEquipes"></canvas>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script>
    new Chart(document.getElementById('chartEquipes'), {{
      type: 'bar',
      data: {{
        labels: {equipes},
        datasets: [{{
          label: 'Productivité (%)',
          data: {prod_vals},
          backgroundColor: {colors},
          borderRadius: 4,
        }}]
      }},
      options: {{
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{
            callbacks: {{
              label: ctx => ctx.raw + '%'
            }}
          }}
        }},
        scales: {{
          x: {{ min: 0, max: 100, ticks: {{ callback: v => v + '%' }} }}
        }}
      }}
    }});
    </script>
    """
    st.components.v1.html(html, height=max(270, len(equipes) * 42))


def _render_equipe_mois(result: ProductiviteResult) -> None:
    """Tableau équipe × mois avec couleurs."""
    st.markdown("### 📅 Équipe × Mois")

    df = result.par_equipe_mois
    if df.empty:
        st.info("Pas de données.")
        return

    pivot = df.pivot_table(
        index="equipe", columns="mois", values="productivite", aggfunc="first"
    ).fillna(0)

    # Formate en pourcentage avec style conditionnel
    def _style_cell(v):
        c = _couleur_perf(v)
        txt = "white" if v >= SEUIL_EXCELLENT or v < SEUIL_FAIBLE else "black"
        return f"background-color:{c};color:{txt};text-align:center"

    styled = (
        pivot.style
        .format("{:.0%}")
        .applymap(_style_cell)
    )
    st.dataframe(styled, use_container_width=True)


def _render_matrice_heatmap(result: ProductiviteResult) -> None:
    """Matrice heatmap technicien × mois avec filtre équipe."""
    st.markdown("### 🔥 Matrice Technicien × Mois")

    df = result.par_tech_mois
    if df.empty:
        st.info("Pas de données.")
        return

    # Filtre équipe
    equipes_dispo = sorted(df["equipe"].unique().tolist())
    equipe_sel = st.selectbox(
        "Filtrer par équipe",
        options=["Toutes les équipes"] + equipes_dispo,
        key="prod_equipe_filter",
    )

    if equipe_sel != "Toutes les équipes":
        df = df[df["equipe"] == equipe_sel]

    if df.empty:
        st.info("Aucun technicien pour cette équipe.")
        return

    pivot = df.pivot_table(
        index="technicien", columns="mois", values="productivite", aggfunc="first"
    ).fillna(0)

    # Tri par productivité moyenne décroissante
    pivot["_moy"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("_moy", ascending=False).drop(columns="_moy")

    def _style_heatmap(v):
        c = _couleur_perf(v)
        txt = "white" if v >= SEUIL_EXCELLENT or v < SEUIL_FAIBLE else "black"
        return f"background-color:{c};color:{txt};text-align:center;font-weight:bold"

    styled = (
        pivot.style
        .format("{:.0%}")
        .applymap(_style_heatmap)
    )
    st.dataframe(styled, use_container_width=True, height=min(600, (len(pivot) + 1) * 36))

    # Légende
    st.markdown(
        f"""
        <div style="display:flex;gap:16px;margin-top:4px;font-size:12px">
          <span style="color:{_COULEUR_EXCELLENT}">■ Excellent ≥ {SEUIL_EXCELLENT:.0%}</span>
          <span style="color:{_COULEUR_BON}">■ Bon ≥ {SEUIL_BON:.0%}</span>
          <span style="color:{_COULEUR_FAIBLE}">■ Faible ≥ {SEUIL_FAIBLE:.0%}</span>
          <span style="color:{_COULEUR_CRITIQUE}">■ Critique &lt; {SEUIL_FAIBLE:.0%}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_tableau_techniciens(result: ProductiviteResult) -> None:
    """Tableau détaillé techniciens avec recherche et tri."""
    st.markdown("### 👷 Détail Techniciens")

    df = result.par_technicien.copy()
    if df.empty:
        st.info("Pas de données techniciens.")
        return

    # Filtres
    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
    with col_f1:
        search = st.text_input("🔍 Rechercher un technicien", key="prod_search_tech")
    with col_f2:
        equipes_dispo = ["Toutes"] + sorted(df["equipe"].unique().tolist())
        eq_filter = st.selectbox("Équipe", equipes_dispo, key="prod_table_equipe")
    with col_f3:
        perf_filter = st.selectbox(
            "Performance",
            ["Tous", "Excellent", "Bon", "Faible", "Critique"],
            key="prod_perf_filter",
        )

    if search:
        df = df[df["technicien"].str.contains(search, case=False, na=False)]
    if eq_filter != "Toutes":
        df = df[df["equipe"] == eq_filter]
    if perf_filter != "Tous":
        df = df[df["perf_label"] == perf_filter]

    # Affichage
    df_display = df[[
        "technicien", "equipe", "facturable", "hr_totale", "productivite", "perf_label", "nb_jours"
    ]].copy()
    df_display = df_display.rename(columns={
        "technicien": "Technicien",
        "equipe": "Équipe",
        "facturable": "Fact. (h)",
        "hr_totale": "Total (h)",
        "productivite": "Productivité",
        "perf_label": "Performance",
        "nb_jours": "Jours",
    })

    def _color_perf(val):
        mapping = {
            "Excellent": f"background-color:{_COULEUR_EXCELLENT};color:white",
            "Bon":       f"background-color:{_COULEUR_BON};color:white",
            "Faible":    f"background-color:{_COULEUR_FAIBLE};color:white",
            "Critique":  f"background-color:{_COULEUR_CRITIQUE};color:white",
        }
        return mapping.get(val, "")

    styled = (
        df_display.style
        .format({"Fact. (h)": "{:.1f}", "Total (h)": "{:.1f}", "Productivité": "{:.1%}"})
        .applymap(_color_perf, subset=["Performance"])
    )

    st.dataframe(styled, use_container_width=True, height=400)
    st.caption(f"{len(df_display)} technicien(s) affiché(s)")

    # Export CSV
    csv = df_display.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="⬇️ Exporter CSV",
        data=csv,
        file_name="productivite_techniciens.csv",
        mime="text/csv",
        key="prod_export_csv",
    )
