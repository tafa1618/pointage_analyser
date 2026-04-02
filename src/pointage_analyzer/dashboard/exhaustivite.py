"""
Onglet Exhaustivité — Calendrier de présence technicien × jour.
Optimisé pour l'export Excel multi-équipes.
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO

from pointage_analyzer.pipeline.exhaustivite_builder import (
    ExhaustiviteBuilder,
    ExhaustiviteError,
    PresenceStatus,
)
from pointage_analyzer.core.config import ScoringConfig

# Palette couleurs sémantiques (RGB Plotly)
STATUS_COLORS = {
    PresenceStatus.PRESENT:      "#2ECC71",
    PresenceStatus.ABSENT:       "#E74C3C",
    PresenceStatus.EXCESSIF:     "#E67E22",
    PresenceStatus.WEEKEND:      "#BDC3C7",
    PresenceStatus.FERIE:        "#85C1E9",
    PresenceStatus.NON_CONCERNE: "#FFFFFF",
}

STATUS_LABELS = {
    PresenceStatus.PRESENT:   "Présent (≤ 8h)",
    PresenceStatus.ABSENT:    "Absent (0h)",
    PresenceStatus.EXCESSIF:  "Excessif (> 8h)",
    PresenceStatus.WEEKEND:   "Week-end",
    PresenceStatus.FERIE:     "Jour férié",
}

def render_exhaustivite_tab(df_presence: pd.DataFrame, config: ScoringConfig) -> None:
    st.header("📅 Contrôle d'Exhaustivité — Calendrier de Présence")

    if df_presence.empty:
        st.error("Données de présence indisponibles. Vérifier le fichier Pointage.")
        return

    builder = ExhaustiviteBuilder(config=config)

    # ── Filtres ───────────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])

    with col_f1:
        equipes_dispo = builder.get_equipes_list(df_presence)
        equipe_filter = st.multiselect(
            "🏢 Équipe(s)",
            options=equipes_dispo,
            default=equipes_dispo[:1] if equipes_dispo else [],
        )

    with col_f2:
        mois_dispo = builder.get_mois_list(df_presence)
        mois_sel = st.selectbox(
            "📆 Mois",
            options=["Tous"] + mois_dispo,
            index=len(mois_dispo) if mois_dispo else 0,
        )

    with col_f3:
        vue_mode = st.radio("Vue", options=["Individuelle", "Équipe"], horizontal=True)

    # ── Filtrage ──────────────────────────────────────────────────────
    df_filtered = builder.get_filtered_matrix(
        df_presence,
        equipe_filter=equipe_filter if equipe_filter else None,
        mois_label=mois_sel if mois_sel != "Tous" else None,
    )

    if df_filtered.empty:
        st.warning("Aucune donnée pour cette sélection.")
        return

    # ── Métriques journalières ────────────────────────────────────────
    _render_metrics(df_filtered, builder)

    st.markdown("---")

    # ── Affichage Calendrier ──────────────────────────────────────────
    if mois_sel == "Tous":
        st.info("💡 Sélectionnez un mois spécifique pour le calendrier détaillé.")
        _render_monthly_summary_chart(df_filtered, builder.compute_daily_stats(df_filtered))
    else:
        try:
            use_nom = vue_mode == "Individuelle"
            pivot_heures, status_matrix = builder.build_pivot_calendar(df_filtered, use_nom=use_nom)
            _render_heatmap_calendar(pivot_heures, status_matrix)
        except Exception as exc:
            st.error(f"Erreur d'affichage : {exc}")

    st.markdown("---")
    _render_export_section(df_presence, builder)

def _render_metrics(df_filtered, builder):
    daily_stats = builder.compute_daily_stats(df_filtered)
    if not daily_stats.empty:
        ouvrable = daily_stats[~daily_stats["est_weekend"]]
        if not ouvrable.empty:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Taux de présence", f"{ouvrable['taux_presence'].mean():.1%}")
            m2.metric("Absences total", int(ouvrable["nb_absents"].sum()))
            m3.metric("Jours > 8h", int(ouvrable["nb_excessifs"].sum()))
            m4.metric("Effectif", df_filtered['salarie_nom'].nunique())

def _render_heatmap_calendar(pivot_heures, status_matrix):
    # Logique Plotly Heatmap (inchangée mais encapsulée)
    dates = [str(d.date()) if hasattr(d, "date") else str(d) for d in pivot_heures.columns]
    techniciens = list(pivot_heures.index)
    
    # Mapping numérique pour les couleurs
    map_val = {PresenceStatus.PRESENT: 1, PresenceStatus.ABSENT: 0, PresenceStatus.EXCESSIF: 2, 
               PresenceStatus.WEEKEND: -1, PresenceStatus.FERIE: -2, PresenceStatus.NON_CONCERNE: -3}

    z = [[map_val.get(status_matrix.loc[t, d], -3) for d in pivot_heures.columns] for t in techniciens]
    
    fig = go.Figure(data=go.Heatmap(
        z=z, x=dates, y=techniciens,
        colorscale=[[0, "#FFFFFF"], [0.2, "#85C1E9"], [0.4, "#BDC3C7"], [0.6, "#E74C3C"], [0.8, "#2ECC71"], [1, "#E67E22"]],
        showscale=False, xgap=1, ygap=1
    ))
    fig.update_layout(height=max(400, 25 * len(techniciens) + 100), margin=dict(l=200, r=20, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# EXPORT EXCEL (VERSION CORRIGÉE)
# ══════════════════════════════════════════════════════════════════════

def _render_export_section(df_presence: pd.DataFrame, builder: ExhaustiviteBuilder) -> None:
    st.subheader("📥 Export Suivi de Présence")
    
    col1, col2 = st.columns(2)
    with col1:
        equipes = st.multiselect("Équipes", builder.get_equipes_list(df_presence), key="exp_eq")
    with col2:
        periodes = ["Toute la période"] + _get_trimestres(df_presence) + builder.get_mois_list(df_presence)
        periode = st.selectbox("Période", periodes, key="exp_per")

    if st.button("⬇️ Générer l'Excel", type="primary"):
        if not equipes:
            st.warning("Choisissez au moins une équipe.")
            return
            
        with st.spinner("Construction du fichier..."):
            try:
                data = _build_excel_engine(df_presence, equipes, periode)
                st.download_button(
                    label="📄 Télécharger le fichier",
                    data=data,
                    file_name=f"Presence_{periode.replace(' ', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Erreur lors de la génération : {e}")

def _build_excel_engine(df_presence: pd.DataFrame, equipes: list[str], periode: str) -> bytes:
    from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    df = _filter_for_export(df_presence, equipes, periode)
    output = BytesIO()

    # Styles
    COLORS = {"NAVY": "002060", "BLEU": "DDEBF7", "VERT": "C6EFCE", "ROUGE": "FFC7CE", "GRIS": "D9D9D9", "JAUNE": "FFF2CC"}
    border = Border(left=Side(style="thin", color="CCCCCC"), right=Side(style="thin", color="CCCCCC"), 
                    top=Side(style="thin", color="CCCCCC"), bottom=Side(style="thin", color="CCCCCC"))

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for equipe in sorted(equipes):
            df_eq = df[df["equipe_nom"] == equipe].copy()
            if df_eq.empty: continue

            # Préparation des dates
            df_eq["date"] = pd.to_datetime(df_eq["date"]).dt.normalize()
            tous_jours = pd.date_range(df_eq["date"].min(), df_eq["date"].max(), freq="D")
            
            # Pivot
            hr_col = "h_totale" if "h_totale" in df_eq.columns else "hr_totale"
            pivot = df_eq.pivot_table(index="salarie_nom", columns="date", values=hr_col, aggfunc="sum")
            pivot = pivot.reindex(columns=tous_jours, fill_value=0).fillna(0)
            
            ws = writer.book.create_sheet(title=str(equipe)[:31])
            
            # Header
            ws["A1"] = f"SUIVI PRÉSENCE - {equipe} ({periode})"
            ws["A1"].font = Font(bold=True, color="FFFFFF", size=12)
            ws["A1"].fill = PatternFill("solid", fgColor=COLORS["NAVY"])
            
            # En-têtes colonnes
            ws.cell(row=3, column=1, value="Technicien").fill = PatternFill("solid", fgColor=COLORS["NAVY"])
            ws.cell(row=3, column=1).font = Font(bold=True, color="FFFFFF")

            for i, jour in enumerate(tous_jours, start=2):
                cell = ws.cell(row=3, column=i, value=f"{jour.strftime('%d/%m')}\n{jour.strftime('%a')}")
                cell.fill = PatternFill("solid", fgColor=COLORS["BLEU"])
                cell.alignment = Alignment(wrap_text=True, horizontal="center")
                cell.border = border
                ws.column_dimensions[get_column_letter(i)].width = 7

            # Données
            techniciens = list(pivot.index)
            nb_jours_ouvres = sum(1 for d in tous_jours if d.weekday() < 5)

            for r_idx, tech in enumerate(techniciens, start=4):
                ws.cell(row=r_idx, column=1, value=tech).border = border
                presents = 0
                
                for c_idx, jour in enumerate(tous_jours, start=2):
                    cell = ws.cell(row=r_idx, column=c_idx)
                    val = pivot.loc[tech, jour]
                    is_weekend = jour.weekday() >= 5
                    
                    if is_weekend:
                        cell.fill = PatternFill("solid", fgColor=COLORS["GRIS"])
                    elif val > 0:
                        cell.value = "P"
                        cell.fill = PatternFill("solid", fgColor=COLORS["VERT"])
                        presents += 1
                    else:
                        cell.fill = PatternFill("solid", fgColor=COLORS["ROUGE"])
                    cell.border = border

                # Résumé
                res_col = len(tous_jours) + 2
                tx = presents / nb_jours_ouvres if nb_jours_ouvres > 0 else 0
                for i, v in enumerate([presents, nb_jours_ouvres, tx]):
                    c = ws.cell(row=r_idx, column=res_col + i, value=v)
                    c.border = border
                    if i == 2: c.number_format = '0%'
                    else: c.fill = PatternFill("solid", fgColor=COLORS["JAUNE"])

            ws.freeze_panes = "B4"
            ws.column_dimensions["A"].width = 30

    return output.getvalue()

def _get_trimestres(df):
    if "date" not in df.columns: return []
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    return sorted(list(set([f"T{(d.month-1)//3 + 1} {d.year}" for d in dates])))

def _filter_for_export(df_presence, equipes, periode):
    df = df_presence.copy()
    df["date"] = pd.to_datetime(df["date"])
    if equipes: df = df[df["equipe_nom"].isin(equipes)]
    
    if periode != "Toute la période":
        if periode.startswith("T"):
            q, y = int(periode[1]), int(periode[3:])
            df = df[(df["date"].dt.year == y) & (((df["date"].dt.month-1)//3 + 1) == q)]
        else:
            df = df[df["date"].dt.to_period("M").astype(str) == periode]
    return df

def _render_monthly_summary_chart(df, daily_stats):
    if daily_stats.empty: return
    import plotly.express as px
    fig = px.bar(daily_stats[~daily_stats["est_weekend"]], x="date", y="taux_presence", 
                 title="Taux de présence moyen", color_discrete_sequence=["#2ECC71"])
    st.plotly_chart(fig, use_container_width=True)
