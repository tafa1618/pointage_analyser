"""
Onglet Productivité — dashboard Streamlit.

Consomme un ProductiviteResult produit par ProductiviteBuilder.
Aucun calcul métier ici — uniquement affichage.

Formule affichée : Facturable / (Facturable + Non Facturable)
Les heures Allouées sont exclues du dénominateur.

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

try:
    from pointage_analyzer.pipeline.productivite_builder import (
        ProductiviteResult,
        SEUIL_EXCELLENT,
        SEUIL_BON,
        SEUIL_FAIBLE,
    )
except ImportError:
    from productivite_builder import (  # type: ignore[no-redef]
        ProductiviteResult,
        SEUIL_EXCELLENT,
        SEUIL_BON,
        SEUIL_FAIBLE,
    )

# ─── Couleurs ────────────────────────────────────────────────────────────────
_COULEUR_EXCELLENT = "#28a745"
_COULEUR_BON       = "#fd7e14"
_COULEUR_FAIBLE    = "#dc3545"
_COULEUR_CRITIQUE  = "#6c757d"
_COULEUR_PRIMAIRE  = "#002060"
_COULEUR_ACCENT    = "#FFCD11"


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
    """Point d'entrée appelé depuis app.py."""
    if result is None or (result.ytd_facturable + result.ytd_non_facturable) == 0:
        st.info("Aucune donnée de productivité disponible. Lancez l'analyse d'abord.")
        return

    st.markdown("## ⚡ Productivité Techniciens")
    st.caption(
        f"Période : **{result.periode_debut}** → **{result.periode_fin}** | "
        f"{result.nb_techniciens} techniciens | {result.nb_equipes} équipes"
    )

    _render_kpi_cards(result)
    st.divider()
    _render_evolution_mensuelle(result)
    st.divider()

    col_eq, col_eq_mois = st.columns([1, 1])
    with col_eq:
        _render_barres_equipes(result)
    with col_eq_mois:
        _render_equipe_mois(result)

    st.divider()
    _render_matrice_heatmap(result)
    st.divider()
    _render_tableau_techniciens(result)
    st.divider()
    _render_simulateur_global(result)
    st.divider()
    _render_analyse_proxy(result)


# ─────────────────────────────────────────────────────────────────────────────
# Composants internes
# ─────────────────────────────────────────────────────────────────────────────

def _render_kpi_cards(result: ProductiviteResult) -> None:
    """4 métriques globales YTD."""
    prod    = result.ytd_productivite
    couleur = _couleur_perf(prod)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            label="🎯 Productivité YTD",
            value=_pct(prod),
            help="Σ Facturable / (Σ Facturable + Σ Non Facturable) — heures Allouées exclues",
        )
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
            label="📉 Heures Non Fact.",
            value=f"{result.ytd_non_facturable:,.0f}h",
            help="Heures non facturables uniquement (Allouées exclues)",
        )

    with c4:
        st.metric(
            label="📋 Heures Allouées",
            value=f"{result.ytd_allouee:,.0f}h",
            help="Formations, déplacements, réunions — exclues du calcul de productivité",
        )


def _render_evolution_mensuelle(result: ProductiviteResult) -> None:
    """Courbe d'évolution mensuelle."""
    st.markdown("### 📈 Évolution mensuelle")

    df = result.par_mois
    if df.empty:
        st.info("Pas de données mensuelles.")
        return

    mois_labels  = df["mois"].tolist()
    prod_values  = [round(v * 100, 1) for v in df["productivite"].tolist()]
    fact_values  = [round(v, 1) for v in df["facturable"].tolist()]
    nonfact_values = [round(v, 1) for v in df["non_facturable"].tolist()]

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
            label: 'Heures Non Facturables',
            data: {nonfact_values},
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

    def _style_cell(v):
        c   = _couleur_perf(v)
        txt = "white" if v >= SEUIL_EXCELLENT or v < SEUIL_FAIBLE else "black"
        return f"background-color:{c};color:{txt};text-align:center"

    styled = pivot.style.format("{:.0%}").applymap(_style_cell)
    st.dataframe(styled, use_container_width=True)


def _render_matrice_heatmap(result: ProductiviteResult) -> None:
    """Matrice heatmap technicien × mois avec filtre équipe."""
    st.markdown("### 🔥 Matrice Technicien × Mois")

    df = result.par_tech_mois
    if df.empty:
        st.info("Pas de données.")
        return

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

    pivot["_moy"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("_moy", ascending=False).drop(columns="_moy")

    def _style_heatmap(v):
        c   = _couleur_perf(v)
        txt = "white" if v >= SEUIL_EXCELLENT or v < SEUIL_FAIBLE else "black"
        return f"background-color:{c};color:{txt};text-align:center;font-weight:bold"

    styled = pivot.style.format("{:.0%}").applymap(_style_heatmap)
    st.dataframe(styled, use_container_width=True, height=min(600, (len(pivot) + 1) * 36))

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

    # Dénominateur réel = Facturable + Non Facturable (sans Allouées)
    df["base_calcul"] = df["facturable"] + df["non_facturable"]

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

    # Affichage — colonnes claires, dénominateur explicite
    df_display = df[[
        "technicien", "equipe",
        "facturable", "non_facturable", "base_calcul",
        "productivite", "perf_label", "nb_jours",
    ]].copy()

    df_display = df_display.rename(columns={
        "technicien":    "Technicien",
        "equipe":        "Équipe",
        "facturable":    "Fact. (h)",
        "non_facturable": "Non Fact. (h)",
        "base_calcul":   "Base calcul (h)",
        "productivite":  "Productivité",
        "perf_label":    "Performance",
        "nb_jours":      "Jours",
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
        .format({
            "Fact. (h)":      "{:.1f}",
            "Non Fact. (h)":  "{:.1f}",
            "Base calcul (h)": "{:.1f}",
            "Productivité":   "{:.1%}",
        })
        .applymap(_color_perf, subset=["Performance"])
    )

    st.dataframe(styled, use_container_width=True, height=400)
    st.caption(
        f"{len(df_display)} technicien(s) affiché(s) — "
        f"Productivité = Fact. / Base calcul (heures Allouées exclues)"
    )

    csv = df_display.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="⬇️ Exporter CSV",
        data=csv,
        file_name="productivite_techniciens.csv",
        mime="text/csv",
        key="prod_export_csv",
    )

def _render_simulateur_global(result: ProductiviteResult) -> None:
    """
    Section 6 — Simulateur productivité globale dynamique.
    Sélection d'équipes → recalcul du KPI global en temps réel.
    """
    st.markdown("### 🎛️ Simulateur Productivité Globale")
    st.caption(
        "Sélectionnez les équipes à inclure dans le calcul — "
        "utile pour exclure le CRC ou toute autre équipe du périmètre."
    )

    df_eq = result.par_equipe.copy()
    if df_eq.empty:
        st.info("Pas de données équipes.")
        return

    equipes_toutes = sorted(df_eq["equipe"].tolist())

    # Identifier les équipes CRC connues par défaut
    crc_keywords = ["remontage transmission", "crc"]
    crc_default  = [
        e for e in equipes_toutes
        if any(k in e.lower() for k in crc_keywords)
    ]

    col_sel, col_kpi = st.columns([2, 1])

    with col_sel:
        equipes_sel = st.multiselect(
            "Équipes incluses dans le calcul",
            options=equipes_toutes,
            default=[e for e in equipes_toutes if e not in crc_default],
            key="prod_sim_equipes",
        )

    # Recalcul dynamique sur les équipes sélectionnées
    if not equipes_sel:
        with col_kpi:
            st.warning("Sélectionnez au moins une équipe.")
        return

    df_sel      = df_eq[df_eq["equipe"].isin(equipes_sel)]
    f_sel       = df_sel["facturable"].sum()
    nf_sel      = df_sel["non_facturable"].sum()
    denom_sel   = f_sel + nf_sel
    prod_sel    = f_sel / denom_sel if denom_sel > 0 else 0.0

    # Productivité de référence (toutes équipes)
    f_all     = df_eq["facturable"].sum()
    nf_all    = df_eq["non_facturable"].sum()
    prod_all  = f_all / (f_all + nf_all) if (f_all + nf_all) > 0 else 0.0
    delta     = prod_sel - prod_all

    with col_kpi:
        st.metric(
            label="🎯 Productivité périmètre sélectionné",
            value=_pct(prod_sel),
            delta=f"{delta:+.1%} vs toutes équipes",
        )

    # Détail des équipes incluses / exclues
    df_eq["statut"] = df_eq["equipe"].apply(
        lambda e: "✅ Incluse" if e in equipes_sel else "❌ Exclue"
    )
    df_eq["contribution"] = (
        df_eq["facturable"] / df_eq["non_facturable"].add(df_eq["facturable"]).replace(0, float("nan"))
    ).fillna(0)

    df_sim_display = df_eq[[
        "statut", "equipe", "facturable", "non_facturable", "productivite"
    ]].copy().rename(columns={
        "statut":        "Statut",
        "equipe":        "Équipe",
        "facturable":    "Fact. (h)",
        "non_facturable": "Non Fact. (h)",
        "productivite":  "Productivité",
    })

    def _color_statut(val):
        if val == "✅ Incluse":
            return "background-color:#d4edda;color:#155724"
        return "background-color:#f8d7da;color:#721c24"

    styled = (
        df_sim_display.style
        .format({"Fact. (h)": "{:.1f}", "Non Fact. (h)": "{:.1f}", "Productivité": "{:.1%}"})
        .applymap(_color_statut, subset=["Statut"])
    )
    st.dataframe(styled, use_container_width=True, height=350)


def _render_analyse_proxy(result: ProductiviteResult) -> None:
    """
    Section 7 — Analyse corrélation & impact équipe proxy.
    Corrélation productivité équipe vs global + impact si retrait.
    """
    import numpy as np

    st.markdown("### 🔬 Analyse Équipe Proxy")
    st.caption(
        "Quelle équipe tire (ou plombe) la productivité globale ? "
        "Corrélation mensuelle + impact simulé si retrait."
    )

    df_em  = result.par_equipe_mois.copy()   # equipe × mois
    df_eq  = result.par_equipe.copy()         # equipe YTD
    df_m   = result.par_mois.copy()           # global × mois

    if df_em.empty or df_m.empty:
        st.info("Pas assez de données pour l'analyse proxy.")
        return

    # ── Corrélation mensuelle équipe vs global ────────────────────────
    pivot = df_em.pivot_table(
        index="mois", columns="equipe", values="productivite"
    ).fillna(0)
    pivot = pivot.join(df_m.set_index("mois")["productivite"].rename("_global"))

    corr_series = pivot.corr()["_global"].drop("_global").sort_values(ascending=False)
    corr_df = corr_series.reset_index()
    corr_df.columns = ["equipe", "correlation"]
    corr_df = corr_df.dropna()

    # ── Impact retrait équipe sur prod YTD ───────────────────────────
    f_all  = df_eq["facturable"].sum()
    nf_all = df_eq["non_facturable"].sum()
    prod_all = f_all / (f_all + nf_all) if (f_all + nf_all) > 0 else 0.0

    impacts = []
    for _, row in df_eq.iterrows():
        f_sans  = f_all  - row["facturable"]
        nf_sans = nf_all - row["non_facturable"]
        prod_sans = f_sans / (f_sans + nf_sans) if (f_sans + nf_sans) > 0 else 0.0
        impacts.append({
            "equipe":     row["equipe"],
            "prod_sans":  prod_sans,
            "delta":      prod_sans - prod_all,
        })

    impact_df = pd.DataFrame(impacts).sort_values("delta", ascending=False)

    # ── Fusion corrélation + impact ───────────────────────────────────
    proxy_df = corr_df.merge(impact_df, on="equipe", how="outer")
    proxy_df["prod_ytd"] = proxy_df["equipe"].map(
        df_eq.set_index("equipe")["productivite"]
    )

    # ── Affichage côte à côte ─────────────────────────────────────────
    col_corr, col_impact = st.columns(2)

    with col_corr:
        st.markdown("#### 📊 Corrélation vs productivité globale")
        st.caption("Calculée sur les mois disponibles (mensuel)")

        if corr_df.empty:
            st.info("Pas assez de mois pour calculer une corrélation.")
        else:
            corr_sorted = corr_df.sort_values("correlation", ascending=True)
            equipes_c   = corr_sorted["equipe"].tolist()
            corr_vals   = [round(v, 3) for v in corr_sorted["correlation"].tolist()]
            colors_c    = [
                "rgba(40,167,69,0.7)"  if v >= 0.7  else
                "rgba(253,126,20,0.7)" if v >= 0.3  else
                "rgba(220,53,69,0.7)"
                for v in corr_vals
            ]

            html_corr = f"""
            <div style="position:relative;height:{max(250, len(equipes_c)*38)}px">
              <canvas id="chartCorr"></canvas>
            </div>
            <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
            <script>
            new Chart(document.getElementById('chartCorr'), {{
              type: 'bar',
              data: {{
                labels: {equipes_c},
                datasets: [{{
                  label: 'Corrélation',
                  data: {corr_vals},
                  backgroundColor: {colors_c},
                  borderRadius: 4,
                }}]
              }},
              options: {{
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                  x: {{ min: -1, max: 1,
                         ticks: {{ callback: v => v.toFixed(1) }} }}
                }}
              }}
            }});
            </script>
            """
            st.components.v1.html(html_corr, height=max(270, len(equipes_c) * 40))

    with col_impact:
        st.markdown("#### ⚖️ Impact si équipe retirée du périmètre")
        st.caption(f"Référence : productivité globale YTD = {_pct(prod_all)}")

        impact_sorted = impact_df.sort_values("delta", ascending=True)
        equipes_i     = impact_sorted["equipe"].tolist()
        delta_vals    = [round(v * 100, 2) for v in impact_sorted["delta"].tolist()]
        colors_i      = [
            "rgba(40,167,69,0.7)"  if v > 0 else
            "rgba(220,53,69,0.7)"
            for v in delta_vals
        ]

        html_impact = f"""
        <div style="position:relative;height:{max(250, len(equipes_i)*38)}px">
          <canvas id="chartImpact"></canvas>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
        <script>
        new Chart(document.getElementById('chartImpact'), {{
          type: 'bar',
          data: {{
            labels: {equipes_i},
            datasets: [{{
              label: 'Delta productivité (%)',
              data: {delta_vals},
              backgroundColor: {colors_i},
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
                  label: ctx => (ctx.raw > 0 ? '+' : '') + ctx.raw + '%'
                }}
              }}
            }},
            scales: {{
              x: {{
                ticks: {{ callback: v => (v > 0 ? '+' : '') + v + '%' }}
              }}
            }}
          }}
        }});
        </script>
        """
        st.components.v1.html(html_impact, height=max(270, len(equipes_i) * 40))

    # ── Tableau récapitulatif proxy ───────────────────────────────────
    st.markdown("#### 📋 Tableau récapitulatif")

    proxy_display = proxy_df[[
        "equipe", "prod_ytd", "correlation", "delta"
    ]].copy().rename(columns={
        "equipe":      "Équipe",
        "prod_ytd":    "Prod. YTD",
        "correlation": "Corrélation vs global",
        "delta":       "Impact si retrait",
    }).sort_values("Impact si retrait", ascending=True)

    def _color_delta(val):
        if pd.isna(val):
            return ""
        if val > 0.02:
            return f"background-color:{_COULEUR_CRITIQUE};color:white"   # retire → améliore → équipe plombe
        elif val > 0:
            return "background-color:#fff3cd;color:#856404"
        elif val < -0.02:
            return f"background-color:{_COULEUR_EXCELLENT};color:white"  # retire → détériore → équipe tire
        return ""

    def _color_corr(val):
        if pd.isna(val):
            return ""
        if val >= 0.7:
            return f"background-color:{_COULEUR_EXCELLENT};color:white"
        elif val >= 0.3:
            return "background-color:#fff3cd;color:#856404"
        elif val < 0:
            return f"background-color:{_COULEUR_FAIBLE};color:white"
        return ""

    styled_proxy = (
        proxy_display.style
        .format({
            "Prod. YTD":            "{:.1%}",
            "Corrélation vs global": lambda v: f"{v:.2f}" if pd.notna(v) else "—",
            "Impact si retrait":     lambda v: f"{v:+.1%}" if pd.notna(v) else "—",
        })
        .applymap(_color_delta, subset=["Impact si retrait"])
        .applymap(_color_corr,  subset=["Corrélation vs global"])
    )
    st.dataframe(styled_proxy, use_container_width=True)

    # ── Interprétation automatique ────────────────────────────────────
    st.markdown("#### 💡 Interprétation")

    # Équipe qui tire le plus (retrait dégrade le plus)
    tire = impact_df.loc[impact_df["delta"].idxmin()]
    # Équipe qui plombe le plus (retrait améliore le plus)
    plombe = impact_df.loc[impact_df["delta"].idxmax()]

    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.success(
            f"**🏆 Équipe locomotive : {tire['equipe']}**\n\n"
            f"Sans elle, la productivité globale baisserait de **{abs(tire['delta']):.1%}** "
            f"({_pct(prod_all)} → {_pct(tire['prod_sans'])})."
        )
    with col_i2:
        st.error(
            f"**⚠️ Équipe qui plombe : {plombe['equipe']}**\n\n"
            f"Sans elle, la productivité globale monterait de **{plombe['delta']:.1%}** "
            f"({_pct(prod_all)} → {_pct(plombe['prod_sans'])})."
        )
