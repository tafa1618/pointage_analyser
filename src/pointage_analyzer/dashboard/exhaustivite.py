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

_JOURS_FR = {0: "Lun", 1: "Mar", 2: "Mer", 3: "Jeu", 4: "Ven", 5: "Sam", 6: "Dim"}


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


def _render_metrics(df_filtered: pd.DataFrame, builder: ExhaustiviteBuilder) -> None:
    daily_stats = builder.compute_daily_stats(df_filtered)
    if not daily_stats.empty:
        ouvrable = daily_stats[~daily_stats["est_weekend"]]
        if not ouvrable.empty:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Taux de présence", f"{ouvrable['taux_presence'].mean():.1%}")
            m2.metric("Absences total", int(ouvrable["nb_absents"].sum()))
            m3.metric("Jours > 8h", int(ouvrable["nb_excessifs"].sum()))
            m4.metric("Effectif", df_filtered["salarie_nom"].nunique())


def _render_heatmap_calendar(pivot_heures: pd.DataFrame, status_matrix: pd.DataFrame) -> None:
    dates       = [str(d.date()) if hasattr(d, "date") else str(d) for d in pivot_heures.columns]
    techniciens = list(pivot_heures.index)

    map_val = {
        PresenceStatus.PRESENT:      1,
        PresenceStatus.ABSENT:       0,
        PresenceStatus.EXCESSIF:     2,
        PresenceStatus.WEEKEND:      -1,
        PresenceStatus.FERIE:        -2,
        PresenceStatus.NON_CONCERNE: -3,
    }

    z = [
        [map_val.get(status_matrix.loc[t, d], -3) for d in pivot_heures.columns]
        for t in techniciens
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z, x=dates, y=techniciens,
        colorscale=[
            [0.0,   STATUS_COLORS[PresenceStatus.NON_CONCERNE]],
            [0.167, STATUS_COLORS[PresenceStatus.FERIE]],
            [0.333, STATUS_COLORS[PresenceStatus.WEEKEND]],
            [0.5,   STATUS_COLORS[PresenceStatus.ABSENT]],
            [0.667, STATUS_COLORS[PresenceStatus.PRESENT]],
            [1.0,   STATUS_COLORS[PresenceStatus.EXCESSIF]],
        ],
        showscale=False, xgap=1, ygap=1,
    ))
    fig.update_layout(
        height=max(400, 25 * len(techniciens) + 100),
        margin=dict(l=200, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Légende
    col_leg = st.columns(len(STATUS_LABELS))
    for i, (status, label) in enumerate(STATUS_LABELS.items()):
        with col_leg[i]:
            color = STATUS_COLORS[status]
            st.markdown(
                f'<div style="background:{color};border-radius:4px;padding:4px 8px;'
                f'text-align:center;font-size:12px;color:{"white" if status == PresenceStatus.ABSENT else "black"}">'
                f'{label}</div>',
                unsafe_allow_html=True,
            )


def _render_monthly_summary_chart(df_filtered: pd.DataFrame, daily_stats: pd.DataFrame) -> None:
    if daily_stats.empty:
        return
    import plotly.express as px
    fig = px.bar(
        daily_stats[~daily_stats["est_weekend"]],
        x="date", y="taux_presence",
        color="taux_presence",
        color_continuous_scale=["#E74C3C", "#F39C12", "#2ECC71"],
        title="Taux de présence journalier (jours ouvrables)",
        labels={"taux_presence": "Taux présence", "date": "Date"},
    )
    fig.update_layout(yaxis_tickformat=".0%", height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# EXPORT EXCEL
# ══════════════════════════════════════════════════════════════════════

def _render_export_section(df_presence: pd.DataFrame, builder: ExhaustiviteBuilder) -> None:
    st.subheader("📥 Export Suivi de Présence")
    st.caption("Une feuille par équipe — P = présent, vide = absent.")

    col1, col2 = st.columns(2)
    with col1:
        equipes = st.multiselect(
            "Équipes à exporter",
            options=builder.get_equipes_list(df_presence),
            default=builder.get_equipes_list(df_presence),
            key="exp_eq",
        )
    with col2:
        periodes = (
            ["Toute la période"]
            + _get_trimestres(df_presence)
            + builder.get_mois_list(df_presence)
        )
        periode = st.selectbox("Période", periodes, key="exp_per")

    if not equipes:
        st.warning("Sélectionnez au moins une équipe.")
        return

    # Aperçu périmètre
    df_prev  = _filter_for_export(df_presence, equipes, periode)
    nb_tech  = df_prev["salarie_nom"].nunique() if "salarie_nom" in df_prev.columns else 0
    nb_jours = df_prev["date"].nunique() if "date" in df_prev.columns else 0
    st.info(
        f"**Périmètre** : {len(equipes)} équipe(s) · "
        f"{nb_tech} technicien(s) · {nb_jours} jour(s) · {periode}"
    )

    if st.button("⬇️ Générer l'Excel", type="primary", key="export_btn"):
        with st.spinner("Construction du fichier…"):
            try:
                data = _build_excel_engine(df_presence, equipes, periode)
                nom  = f"Presence_{periode.replace(' ', '_').replace('/', '-')}.xlsx"
                st.download_button(
                    label="📄 Télécharger",
                    data=data,
                    file_name=nom,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="export_dl",
                )
                st.success(f"✅ {len(equipes)} feuille(s) générée(s)")
            except Exception as exc:
                st.error(f"Erreur : {exc}")


def _build_excel_engine(df_presence: pd.DataFrame, equipes: list[str], periode: str) -> bytes:
    """
    Moteur de génération Excel.
    Lit les valeurs directement depuis le pivot Python — pas de cell.value.
    """
    from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    df = _filter_for_export(df_presence, equipes, periode)
    if df.empty:
        raise ValueError("Aucune donnée pour cette sélection.")

    output = BytesIO()

    NAVY  = "002060"
    BLEU  = "DDEBF7"
    VERT  = "C6EFCE"
    ROUGE = "FFC7CE"
    GRIS  = "D9D9D9"
    JAUNE = "FFF2CC"

    thin   = Side(style="thin", color="CCCCCC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for equipe in sorted(equipes):
            df_eq = df[df["equipe_nom"] == equipe].copy()
            if df_eq.empty:
                continue

            # Dates spécifiques à cette équipe
            df_eq["date"] = pd.to_datetime(df_eq["date"]).dt.normalize()
            tous_jours    = pd.date_range(df_eq["date"].min(), df_eq["date"].max(), freq="D")

            # Pivot heures (valeurs numériques brutes)
            hr_col = "h_totale" if "h_totale" in df_eq.columns else "hr_totale"
            if hr_col not in df_eq.columns:
                df_eq[hr_col] = 0.0

            pivot = df_eq.pivot_table(
                index="salarie_nom", columns="date", values=hr_col, aggfunc="sum"
            ).reindex(columns=tous_jours, fill_value=0).fillna(0)

            techniciens     = list(pivot.index)
            nb_jours_ouvres = sum(1 for d in tous_jours if d.weekday() < 5)

            # Créer la feuille
            ws = writer.book.create_sheet(title=equipe[:31])

            # ── Titre ─────────────────────────────────────────────────
            ws["A1"] = f"SUIVI PRÉSENCE — {equipe} ({periode})"
            ws["A1"].font      = Font(bold=True, color="FFFFFF", size=12)
            ws["A1"].fill      = PatternFill("solid", fgColor=NAVY)
            ws["A1"].alignment = Alignment(horizontal="left")

            # ── En-tête colonne technicien ────────────────────────────
            ws.cell(row=3, column=1).value     = "Technicien"
            ws.cell(row=3, column=1).font      = Font(bold=True, color="FFFFFF", size=10)
            ws.cell(row=3, column=1).fill      = PatternFill("solid", fgColor=NAVY)
            ws.cell(row=3, column=1).alignment = Alignment(horizontal="left")
            ws.cell(row=3, column=1).border    = border
            ws.column_dimensions["A"].width    = 30

            # ── En-têtes colonnes dates ───────────────────────────────
            for c_idx, jour in enumerate(tous_jours, start=2):
                cell             = ws.cell(row=3, column=c_idx)
                cell.value       = f"{jour.strftime('%d/%m')}\n{_JOURS_FR[jour.weekday()]}"
                cell.fill        = PatternFill("solid", fgColor=BLEU)
                cell.font        = Font(bold=True, size=8)
                cell.alignment   = Alignment(wrap_text=True, horizontal="center", vertical="center")
                cell.border      = border
                ws.column_dimensions[get_column_letter(c_idx)].width = 7

            # ── En-têtes colonnes résumé ──────────────────────────────
            res_start = len(tous_jours) + 2
            for i, label in enumerate(["Jours présents", "Jours ouvrés", "Taux présence"]):
                col_idx  = res_start + i
                cell     = ws.cell(row=3, column=col_idx)
                cell.value     = label
                cell.fill      = PatternFill("solid", fgColor="FFC000")
                cell.font      = Font(bold=True, size=9)
                cell.alignment = Alignment(horizontal="center", wrap_text=True)
                cell.border    = border
                ws.column_dimensions[get_column_letter(col_idx)].width = 12

            # ── Données ───────────────────────────────────────────────
            for r_idx, tech in enumerate(techniciens, start=4):

                # Nom technicien
                cell_tech            = ws.cell(row=r_idx, column=1)
                cell_tech.value      = tech
                cell_tech.font       = Font(bold=True, size=9)
                cell_tech.fill       = PatternFill("solid", fgColor="EEF3FA")
                cell_tech.alignment  = Alignment(horizontal="left", vertical="center")
                cell_tech.border     = border
                ws.row_dimensions[r_idx].height = 18

                presents = 0

                for c_idx, jour in enumerate(tous_jours, start=2):
                    cell       = ws.cell(row=r_idx, column=c_idx)
                    cell.alignment = Alignment(horizontal="center")
                    cell.border    = border

                    # ← lecture directe du pivot, jamais de cell.value
                    val        = pivot.loc[tech, jour]
                    is_weekend = jour.weekday() >= 5

                    if is_weekend:
                        cell.fill  = PatternFill("solid", fgColor=GRIS)
                    elif val > 0:
                        cell.value = "P"
                        cell.fill  = PatternFill("solid", fgColor=VERT)
                        cell.font  = Font(bold=True, size=9, color="276221")
                        presents  += 1
                    else:
                        cell.fill  = PatternFill("solid", fgColor=ROUGE)

                # ── Résumé ────────────────────────────────────────────
                taux = round(presents / nb_jours_ouvres, 2) if nb_jours_ouvres > 0 else 0

                for i, (val, label) in enumerate([
                    (presents,        "Jours présents"),
                    (nb_jours_ouvres, "Jours ouvrés"),
                    (taux,            "Taux présence"),
                ]):
                    col_idx        = res_start + i
                    cell           = ws.cell(row=r_idx, column=col_idx)
                    cell.value     = val
                    cell.alignment = Alignment(horizontal="center")
                    cell.border    = border

                    if label == "Taux présence":
                        cell.number_format = "0%"
                        if taux >= 0.9:
                            cell.fill = PatternFill("solid", fgColor="C6EFCE")
                            cell.font = Font(bold=True, color="276221", size=9)
                        elif taux >= 0.7:
                            cell.fill = PatternFill("solid", fgColor="FFEB9C")
                            cell.font = Font(bold=True, color="9C6500", size=9)
                        else:
                            cell.fill = PatternFill("solid", fgColor="FFC7CE")
                            cell.font = Font(bold=True, color="9C0006", size=9)
                    else:
                        cell.fill = PatternFill("solid", fgColor=JAUNE)

            ws.row_dimensions[3].height = 36
            ws.freeze_panes = "B4"

    return output.getvalue()


def _get_trimestres(df: pd.DataFrame) -> list[str]:
    if "date" not in df.columns:
        return []
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    return sorted(set(f"T{(d.month - 1) // 3 + 1} {d.year}" for d in dates))


def _filter_for_export(df_presence: pd.DataFrame, equipes: list[str], periode: str) -> pd.DataFrame:
    df = df_presence.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if equipes and "equipe_nom" in df.columns:
        df = df[df["equipe_nom"].isin(equipes)]

    if periode != "Toute la période":
        if periode.startswith("T"):
            q = int(periode[1])
            y = int(periode.split()[1])
            df = df[
                (df["date"].dt.year == y) &
                (((df["date"].dt.month - 1) // 3 + 1) == q)
            ]
        else:
            df = df[df["date"].dt.to_period("M").astype(str) == periode]

    return df
