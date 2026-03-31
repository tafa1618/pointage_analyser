"""
Onglet Techniciens — Performance composite productivité + efficience.

Deux dimensions complémentaires :
  - Productivité : Facturable / (Facturable + Non Facturable) — tous OR
  - Efficience   : Temps ref / Heures réalisées, clippé [0,1]
                   Temps ref = Temps vendu en priorité, sinon Temps prévu devis
                   Périmètre : OR mono-technicien avec temps_ref > 0 uniquement

Score composite = 0.5 × Productivité + 0.5 × Efficience
Si efficience non calculable → score = productivité seule

Badges :
  🏆 Champion      : score ≥ 0.70
  ✅ Bon           : score ≥ 0.50
  📈 À encourager  : score ≥ 0.30
  🔴 À accompagner : score < 0.30
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

# ── Constantes ────────────────────────────────────────────────────────
_SEUIL_CHAMPION    = 0.70
_SEUIL_BON         = 0.50
_SEUIL_ENCOURAGER  = 0.30

_COULEUR_CHAMPION   = "#002060"   # Navy CAT
_COULEUR_BON        = "#28a745"   # Vert
_COULEUR_ENCOURAGER = "#fd7e14"   # Orange
_COULEUR_ACCOMPAGNER = "#dc3545"  # Rouge

_COULEUR_PROD = "#FFCD11"         # Jaune CAT
_COULEUR_EFF  = "#1f77b4"         # Bleu


def _badge(score: float) -> str:
    if score >= _SEUIL_CHAMPION:   return "🏆 Champion"
    elif score >= _SEUIL_BON:      return "✅ Bon"
    elif score >= _SEUIL_ENCOURAGER: return "📈 À encourager"
    else:                            return "🔴 À accompagner"


def _couleur_badge(badge: str) -> str:
    mapping = {
        "🏆 Champion":      _COULEUR_CHAMPION,
        "✅ Bon":           _COULEUR_BON,
        "📈 À encourager":  _COULEUR_ENCOURAGER,
        "🔴 À accompagner": _COULEUR_ACCOMPAGNER,
    }
    return mapping.get(badge, "#888888")


def _couleur_score(val: float) -> str:
    if val >= _SEUIL_CHAMPION:    return f"background-color:{_COULEUR_CHAMPION};color:white"
    elif val >= _SEUIL_BON:       return f"background-color:{_COULEUR_BON};color:white"
    elif val >= _SEUIL_ENCOURAGER: return f"background-color:{_COULEUR_ENCOURAGER};color:white"
    else:                          return f"background-color:{_COULEUR_ACCOMPAGNER};color:white"


# ── Calculs métier ────────────────────────────────────────────────────

def compute_tech_scores(
    pointage_df: pd.DataFrame,
    bo_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Calcule les scores productivité + efficience par technicien.

    Args:
        pointage_df : DataFrame Pointage harmonisé (colonnes snake_case)
                      OU brut (colonnes françaises) — les deux acceptés
        bo_df       : DataFrame BO harmonisé ou brut, peut être None

    Returns:
        DataFrame technicien-level avec colonnes :
            technicien, equipe, productivite, efficience, score, badge,
            nb_or_tot, nb_or_eff, nb_jours, h_fact, h_nonfact, h_allouee,
            pct_depassement, has_efficience, source_info
    """
    # ── Normalisation colonnes Pointage ──────────────────────────────
    pt = _normalize_pointage(pointage_df)
    if pt.empty:
        return pd.DataFrame()

    pt_valid = pt[pt["_or_id"].notna() & (pt["_or_id"] != "0")].copy()

    # ── Productivité (tous OR, tous techniciens) ──────────────────────
    tech_prod = (
        pt_valid.groupby(["_technom", "_equipe"])
        .agg(
            h_fact=("_facturable", "sum"),
            h_nonfact=("_nonfacturable", "sum"),
            h_allouee=("_allouee", "sum"),
            h_totale=("_hr_totale", "sum"),
            nb_or_tot=("_or_id", "nunique"),
            nb_jours=("_date", "nunique"),
        )
        .reset_index()
    )
    tech_prod["productivite"] = (
        tech_prod["h_fact"] /
        (tech_prod["h_fact"] + tech_prod["h_nonfact"]).replace(0, np.nan)
    ).fillna(0)

    # ── Efficience (OR mono-technicien avec temps_ref > 0) ────────────
    tech_eff = _compute_efficience(pt_valid, bo_df)

    # ── Fusion ───────────────────────────────────────────────────────
    final = tech_prod.merge(tech_eff, on="_technom", how="left")
    final["has_efficience"] = final["efficience"].notna()

    final["score"] = np.where(
        final["has_efficience"],
        0.5 * final["productivite"] + 0.5 * final["efficience"],
        final["productivite"],
    )

    final["badge"] = final["score"].apply(_badge)

    # Renommage colonnes display
    final = final.rename(columns={
        "_technom": "technicien",
        "_equipe":  "equipe",
    })

    return final.sort_values("score", ascending=False).reset_index(drop=True)


def _normalize_pointage(pt: pd.DataFrame) -> pd.DataFrame:
    """Mappe colonnes brutes ou harmonisées vers noms internes _xxx."""
    out = pt.copy()

    def _pick(harm, raw):
        if harm in out.columns: return out[harm]
        if raw  in out.columns: return out[raw]
        return pd.Series(np.nan, index=out.index)

    out["_or_id"]       = _pick("or_id", "OR (Numéro)").astype(str).str.strip()
    out["_technom"]     = _pick("salarie_nom", "Salarié - Nom").fillna("Inconnu")
    out["_equipe"]      = _pick("equipe_nom", "Salarié - Equipe(Nom)").fillna("Inconnu")
    out["_facturable"]  = pd.to_numeric(_pick("facturable", "Facturable"),    errors="coerce").fillna(0)
    out["_nonfacturable"] = pd.to_numeric(_pick("non_facturable", "Non Facturable"), errors="coerce").fillna(0)
    out["_allouee"]     = pd.to_numeric(_pick("allouee", "Allouée"),          errors="coerce").fillna(0)
    out["_hr_totale"]   = pd.to_numeric(_pick("hr_totale", "Hr_Totale"),      errors="coerce").fillna(0)
    out["_date"]        = pd.to_datetime(_pick("date_saisie", "Saisie heures - Date"), errors="coerce")

    out["_or_id"] = out["_or_id"].where(~out["_or_id"].isin(["0", "nan", ""]), np.nan)
    return out


def _compute_efficience(
    pt_valid: pd.DataFrame,
    bo_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Calcule l'efficience sur OR mono-technicien avec temps_ref > 0.
    Temps ref = temps vendu en priorité, sinon temps prévu devis.
    """
    empty = pd.DataFrame(columns=["_technom", "efficience", "nb_or_eff",
                                   "pct_depassement", "nb_or_depasse", "source_info"])

    if bo_df is None or bo_df.empty:
        return empty

    # ── Temps référence depuis BO ─────────────────────────────────────
    bo = bo_df.copy()

    def _pick_bo(harm, raw):
        if harm in bo.columns: return bo[harm]
        if raw  in bo.columns: return bo[raw]
        return pd.Series(0, index=bo.index)

    or_id_col = "or_id" if "or_id" in bo.columns else "N° OR (Segment)"
    bo["_or_id"] = bo[or_id_col].astype(str).str.strip()

    tv = pd.to_numeric(_pick_bo("temps_vendu", "Temps vendu (OR)"),      errors="coerce").fillna(0)
    tp = pd.to_numeric(_pick_bo("temps_prevu_devis", "Temps prévu devis (OR)"), errors="coerce").fillna(0)
    bo["_temps_ref"]    = tv.where(tv > 0, tp)
    bo["_source_ref"]   = np.where(tv > 0, "vendu", np.where(tp > 0, "devis", "aucun"))

    bo_or = (
        bo.groupby("_or_id")
        .agg(
            temps_ref=("_temps_ref", "sum"),
            source_vendu=("_source_ref", lambda x: (x == "vendu").any()),
        )
        .reset_index()
        .rename(columns={"_or_id": "_or_id_bo"})
    )
    bo_or["source_label"] = np.where(bo_or["source_vendu"], "vendu", "devis")

    # ── OR mono-technicien ────────────────────────────────────────────
    nb_tech_par_or = pt_valid.groupby("_or_id")["_technom"].nunique()
    or_mono        = nb_tech_par_or[nb_tech_par_or == 1].index

    pt_mono = pt_valid[pt_valid["_or_id"].isin(or_mono)].copy()
    if pt_mono.empty:
        return empty

    pt_mono_or = (
        pt_mono.groupby(["_technom", "_or_id"])
        .agg(h_realisees=("_hr_totale", "sum"))
        .reset_index()
    )

    pt_mono_or = pt_mono_or.merge(
        bo_or.rename(columns={"_or_id_bo": "_or_id"}),
        on="_or_id", how="left"
    )

    # Éliminer OR sans temps_ref ou sans heures
    pt_mono_or = pt_mono_or[
        (pt_mono_or["temps_ref"] > 0) & (pt_mono_or["h_realisees"] > 0)
    ]
    if pt_mono_or.empty:
        return empty

    pt_mono_or["en_depassement"] = pt_mono_or["h_realisees"] > pt_mono_or["temps_ref"]

    tech_eff = (
        pt_mono_or.groupby("_technom")
        .agg(
            nb_or_eff=("_or_id", "nunique"),
            h_realisees_eff=("h_realisees", "sum"),
            temps_ref_eff=("temps_ref", "sum"),
            nb_or_depasse=("en_depassement", "sum"),
            source_vendu_count=("source_vendu", "sum"),
        )
        .reset_index()
    )

    tech_eff["efficience"] = (
        tech_eff["temps_ref_eff"] / tech_eff["h_realisees_eff"].replace(0, np.nan)
    ).clip(0, 1).fillna(0)

    tech_eff["pct_depassement"] = tech_eff["nb_or_depasse"] / tech_eff["nb_or_eff"]
    tech_eff["source_info"] = np.where(
        tech_eff["source_vendu_count"] > 0, "Temps vendu (priorité)", "Temps prévu devis"
    )

    return tech_eff[["_technom", "efficience", "nb_or_eff",
                      "pct_depassement", "nb_or_depasse", "source_info"]]


# ── Rendu Streamlit ───────────────────────────────────────────────────

def render_techniciens_tab(
    pointage_df: pd.DataFrame,
    bo_df: pd.DataFrame | None = None,
) -> None:
    """
    Point d'entrée appelé depuis app.py.

    Usage :
        from pointage_analyzer.dashboard.techniciens import render_techniciens_tab
        with tab_tech:
            render_techniciens_tab(pt_harm, bo_df)
    """
    st.markdown("## 👷 Performance par Technicien")

    with st.spinner("Calcul des scores en cours…"):
        df = compute_tech_scores(pointage_df, bo_df)

    if df.empty:
        st.info("Aucune donnée disponible.")
        return

    nb_total    = len(df)
    nb_avec_eff = int(df["has_efficience"].sum())
    nb_sans_eff = nb_total - nb_avec_eff

    st.caption(
        f"**{nb_total}** techniciens analysés | "
        f"Score composite (Prod. + Eff.) : **{nb_avec_eff}** techniciens | "
        f"Score productivité seule : **{nb_sans_eff}** (OR multi-techniciens ou sans devis)"
    )

    # ── KPI cards badges ──────────────────────────────────────────────
    _render_badge_kpis(df)
    st.divider()

    # ── Scatter prod vs efficience ────────────────────────────────────
    _render_scatter(df)
    st.divider()

    # ── Tableau ranking ───────────────────────────────────────────────
    _render_ranking_table(df)
    st.divider()

    # ── Note méthodologique ───────────────────────────────────────────
    _render_methodologie(nb_avec_eff, nb_total)


def _render_badge_kpis(df: pd.DataFrame) -> None:
    """4 compteurs de badges."""
    badges = ["🏆 Champion", "✅ Bon", "📈 À encourager", "🔴 À accompagner"]
    counts = {b: (df["badge"] == b).sum() for b in badges}

    c1, c2, c3, c4 = st.columns(4)
    for col, badge_label in zip([c1, c2, c3, c4], badges):
        couleur = _couleur_badge(badge_label)
        with col:
            st.markdown(
                f"""
                <div style="background:{couleur};border-radius:8px;padding:16px;text-align:center">
                  <div style="color:white;font-size:24px;font-weight:bold">{counts[badge_label]}</div>
                  <div style="color:white;font-size:13px;margin-top:4px">{badge_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_scatter(df: pd.DataFrame) -> None:
    """Scatter plot productivité vs efficience."""
    st.markdown("### 🎯 Carte de performance : Productivité vs Efficience")
    st.caption(
        "Chaque point = 1 technicien. "
        "Axe X = Productivité (dépend du business). "
        "Axe Y = Efficience (dépend du technicien). "
        "Points gris = techniciens sans efficience calculable."
    )

    df_eff  = df[df["has_efficience"]].copy()
    df_neff = df[~df["has_efficience"]].copy()

    # Préparer les données pour Chart.js
    def _make_points(sub, badge_col=True):
        points = []
        for _, row in sub.iterrows():
            points.append({
                "x": round(float(row["productivite"]) * 100, 1),
                "y": round(float(row.get("efficience", row["productivite"])) * 100, 1),
                "label": str(row["technicien"]),
                "equipe": str(row["equipe"]),
                "badge": str(row["badge"]) if badge_col else "—",
                "score": round(float(row["score"]) * 100, 1),
                "color": _couleur_badge(str(row["badge"])) if badge_col else "#cccccc",
            })
        return points

    pts_eff  = _make_points(df_eff,  badge_col=True)
    pts_neff = _make_points(df_neff, badge_col=False)

    html = f"""
    <div style="position:relative;height:420px">
      <canvas id="chartScatter"></canvas>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script>
    const ptsEff  = {pts_eff};
    const ptsNeff = {pts_neff};

    new Chart(document.getElementById('chartScatter'), {{
      type: 'scatter',
      data: {{
        datasets: [
          {{
            label: 'Avec efficience',
            data: ptsEff.map(p => ({{x: p.x, y: p.y}})),
            backgroundColor: ptsEff.map(p => p.color + 'CC'),
            borderColor:     ptsEff.map(p => p.color),
            borderWidth: 2,
            pointRadius: 8,
            pointHoverRadius: 11,
          }},
          {{
            label: 'Sans efficience (prod. seule)',
            data: ptsNeff.map(p => ({{x: p.x, y: p.y}})),
            backgroundColor: 'rgba(180,180,180,0.4)',
            borderColor:     '#aaaaaa',
            borderWidth: 1,
            pointRadius: 6,
            pointStyle: 'triangle',
          }},
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
                const allPts = [...ptsEff, ...ptsNeff];
                const p = allPts[ctx.dataIndex + (ctx.datasetIndex === 1 ? ptsEff.length : 0)];
                if (!p) return '';
                return [
                  p.label,
                  'Équipe : ' + p.equipe,
                  'Productivité : ' + p.x + '%',
                  'Efficience : ' + (ctx.datasetIndex === 0 ? p.y + '%' : 'N/A'),
                  'Score : ' + p.score + '%',
                  p.badge,
                ];
              }}
            }}
          }}
        }},
        scales: {{
          x: {{
            title: {{ display: true, text: 'Productivité (%)' }},
            min: 0, max: 105,
            ticks: {{ callback: v => v + '%' }}
          }},
          y: {{
            title: {{ display: true, text: 'Efficience (%)' }},
            min: 0, max: 105,
            ticks: {{ callback: v => v + '%' }}
          }}
        }}
      }}
    }});
    </script>

    <!-- Quadrant labels -->
    <div style="display:flex;gap:24px;margin-top:8px;font-size:12px;color:#666">
      <span>↗️ <b>Haut droite</b> : Productif ET efficient → Champions</span>
      <span>↘️ <b>Bas droite</b> : Productif mais dépasse les devis → à cadrer</span>
      <span>↖️ <b>Haut gauche</b> : Efficient mais peu de facturable → business à développer</span>
      <span>↙️ <b>Bas gauche</b> : À accompagner en priorité</span>
    </div>
    """
    st.components.v1.html(html, height=480)


def _render_ranking_table(df: pd.DataFrame) -> None:
    """Tableau ranking filtrable."""
    st.markdown("### 📋 Ranking Techniciens")

    # Filtres
    col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
    with col_f1:
        search = st.text_input("🔍 Rechercher", key="tech_search")
    with col_f2:
        equipes = ["Toutes"] + sorted(df["equipe"].unique().tolist())
        eq_sel  = st.selectbox("Équipe", equipes, key="tech_equipe")
    with col_f3:
        badges_dispo = ["Tous", "🏆 Champion", "✅ Bon", "📈 À encourager", "🔴 À accompagner"]
        badge_sel    = st.selectbox("Badge", badges_dispo, key="tech_badge")

    dff = df.copy()
    if search:
        dff = dff[dff["technicien"].str.contains(search, case=False, na=False)]
    if eq_sel != "Toutes":
        dff = dff[dff["equipe"] == eq_sel]
    if badge_sel != "Tous":
        dff = dff[dff["badge"] == badge_sel]

    # Colonnes affichées
    dff["Rang"] = range(1, len(dff) + 1)
    dff["Efficience"] = dff["efficience"].apply(
        lambda v: f"{v:.0%}" if pd.notna(v) else "—"
    )
    dff["Dépassements"] = dff.apply(
        lambda r: f"{int(r['nb_or_depasse'])}/{int(r['nb_or_eff'])}" if r["has_efficience"] else "—",
        axis=1
    )

    display = dff[[
        "Rang", "technicien", "equipe",
        "productivite", "Efficience", "score", "badge",
        "nb_or_tot", "Dépassements", "nb_jours"
    ]].rename(columns={
        "technicien":   "Technicien",
        "equipe":       "Équipe",
        "productivite": "Productivité",
        "score":        "Score composite",
        "badge":        "Badge",
        "nb_or_tot":    "OR total",
        "nb_jours":     "Jours",
    })

    def _style_badge(val):
        return f"background-color:{_couleur_badge(val)};color:white;font-weight:bold"

    def _style_score(val):
        return _couleur_score(val)

    styled = (
        display.style
        .format({
            "Productivité":    "{:.0%}",
            "Score composite": "{:.0%}",
        })
        .applymap(_style_badge,  subset=["Badge"])
        .applymap(_style_score,  subset=["Score composite"])
    )

    st.dataframe(styled, use_container_width=True, height=450)
    st.caption(f"{len(display)} technicien(s) affiché(s)")

    csv = display.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Exporter CSV", csv,
        "ranking_techniciens.csv", "text/csv",
        key="tech_export"
    )


def _render_methodologie(nb_avec_eff: int, nb_total: int) -> None:
    """Note méthodologique dépliable."""
    with st.expander("📖 Méthodologie — comment lire ce ranking ?", expanded=False):
        st.markdown(f"""
**Score composite = 50% Productivité + 50% Efficience**

---

**Productivité** *(tous les {nb_total} techniciens)*
> Heures Facturables / (Heures Facturables + Heures Non Facturables)

Mesure la part du temps du technicien qui génère de la valeur facturée.
Cette métrique **dépend du business** (volume d'OR facturables disponibles),
pas uniquement du technicien.

---

**Efficience** *({nb_avec_eff} techniciens sur {nb_total} — OR mono-technicien uniquement)*
> Temps référence / Heures réalisées, clippé à [0, 1]
>
> Temps référence = **Temps vendu** en priorité, sinon **Temps prévu devis**

Mesure si le technicien réalise l'OR dans le temps imparti.
- **Efficience = 1.0** : il a fait aussi vite ou plus vite que prévu ✅
- **Efficience < 1.0** : il a dépassé le temps prévu ⚠️

**Pourquoi OR mono-technicien uniquement ?**
Le temps de devis est au niveau OR entier. S'il y a plusieurs techniciens,
on ne peut pas répartir ce temps entre eux sans biais. On exclut donc ces OR
pour ne garder que des comparaisons rigoureuses.

**Techniciens sans efficience (triangle gris sur le scatter)**
Soit tous leurs OR impliquent plusieurs techniciens, soit leurs OR n'ont pas
de temps de référence renseigné dans BO. Leur score = productivité seule.

---

**Badges**
| Badge | Score composite |
|---|---|
| 🏆 Champion | ≥ 70% |
| ✅ Bon | ≥ 50% |
| 📈 À encourager | ≥ 30% |
| 🔴 À accompagner | < 30% |

> Les Champions sont à **valoriser et fidéliser**.
> Les techniciens À accompagner ont besoin de **support managérial ciblé**,
> pas nécessairement de sanction — le contexte métier (type d'OR, équipe)
> doit toujours être pris en compte.
        """)
