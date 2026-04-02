"""
Onglet Exhaustivité — Calendrier de présence technicien × jour.

Ce module est la couche UI pure de l'exhaustivité.
Toute la logique est dans pipeline/exhaustivite_builder.py.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from pointage_analyzer.pipeline.exhaustivite_builder import (
    ExhaustiviteBuilder,
    ExhaustiviteError,
    PresenceStatus,
)
from pointage_analyzer.core.config import ScoringConfig


# Palette couleurs sémantiques (RGB Plotly)
STATUS_COLORS: dict[str, str] = {
    PresenceStatus.PRESENT:      "#2ECC71",
    PresenceStatus.ABSENT:       "#E74C3C",
    PresenceStatus.EXCESSIF:     "#E67E22",
    PresenceStatus.WEEKEND:      "#BDC3C7",
    PresenceStatus.FERIE:        "#85C1E9",
    PresenceStatus.NON_CONCERNE: "#FFFFFF",
}

STATUS_LABELS: dict[str, str] = {
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
            help="Laisser vide = toutes les équipes",
        )

    with col_f2:
        mois_dispo = builder.get_mois_list(df_presence)
        mois_sel = st.selectbox(
            "📆 Mois",
            options=["Tous"] + mois_dispo,
            index=len(mois_dispo) if mois_dispo else 0,
            help="Sélectionner un mois pour le calendrier",
        )

    with col_f3:
        vue_mode = st.radio(
            "Vue",
            options=["Individuelle", "Équipe"],
            index=0,
            horizontal=True,
        )

    # ── Filtrage ──────────────────────────────────────────────────────
    df_filtered = builder.get_filtered_matrix(
        df_presence,
        equipe_filter=equipe_filter if equipe_filter else None,
        mois_label=mois_sel if mois_sel != "Tous" else None,
    )

    if df_filtered.empty:
        st.warning("Aucune donnée pour cette sélection. Modifier les filtres.")
        return

    # ── Métriques journalières ────────────────────────────────────────
    daily_stats = builder.compute_daily_stats(df_filtered)
    if not daily_stats.empty:
        ouvrable = daily_stats[~daily_stats["est_weekend"]]
        if not ouvrable.empty:
            taux_moy     = ouvrable["taux_presence"].mean()
            nb_abs_total = ouvrable["nb_absents"].sum()
            nb_exc_total = ouvrable["nb_excessifs"].sum()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Taux de présence moyen", f"{taux_moy:.1%}", help="Jours ouvrables uniquement")
            m2.metric("Absences cumulées", f"{nb_abs_total}")
            m3.metric("Jours excessifs (>8h)", f"{nb_exc_total}")
            m4.metric("Techniciens actifs", f"{df_filtered['salarie_nom'].nunique() if 'salarie_nom' in df_filtered.columns else 0}")

    st.markdown("---")

    # ── Calendrier heatmap ────────────────────────────────────────────
    if mois_sel == "Tous":
        st.info("💡 Sélectionner un mois spécifique pour afficher le calendrier détaillé.")
        _render_monthly_summary_chart(df_filtered, daily_stats)
    else:
        try:
            use_nom = vue_mode == "Individuelle"
            pivot_heures, status_matrix = builder.build_pivot_calendar(df_filtered, use_nom=use_nom)
            _render_heatmap_calendar(pivot_heures, status_matrix)
        except ExhaustiviteError as exc:
            st.error(f"Impossible de construire le calendrier: {exc}")
            return

        with st.expander("📊 Données brutes (technicien × jour)"):
            st.dataframe(
                df_filtered.sort_values(["equipe_nom", "salarie_nom", "date"])
                if all(c in df_filtered.columns for c in ["equipe_nom", "salarie_nom", "date"])
                else df_filtered,
                use_container_width=True,
                height=350,
            )

    st.markdown("---")
    _render_export_section(df_presence, builder)


def _render_heatmap_calendar(pivot_heures: pd.DataFrame, status_matrix: pd.DataFrame) -> None:
    dates       = [str(d.date()) if hasattr(d, "date") else str(d) for d in pivot_heures.columns]
    techniciens = list(pivot_heures.index)

    status_to_num = {
        PresenceStatus.PRESENT:      1,
        PresenceStatus.ABSENT:       0,
        PresenceStatus.EXCESSIF:     2,
        PresenceStatus.WEEKEND:      -1,
        PresenceStatus.FERIE:        -2,
        PresenceStatus.NON_CONCERNE: -3,
    }

    z_colors, z_text, customdata = [], [], []
    for tech in techniciens:
        row_colors, row_text, row_custom = [], [], []
        for col_date in pivot_heures.columns:
            heures = pivot_heures.loc[tech, col_date]
            statut = status_matrix.loc[tech, col_date]
            row_colors.append(status_to_num.get(statut, -3))
            row_text.append(f"{heures:.1f}h" if heures > 0 else "")
            row_custom.append([heures, STATUS_LABELS.get(statut, statut)])
        z_colors.append(row_colors)
        z_text.append(row_text)
        customdata.append(row_custom)

    fig = go.Figure(go.Heatmap(
        z=z_colors, x=dates, y=techniciens,
        text=z_text, texttemplate="%{text}",
        textfont={"size": 9, "color": "black"},
        customdata=customdata,
        hovertemplate="<b>%{y}</b><br>%{x}<br>%{customdata[0]:.1f}h — %{customdata[1]}<extra></extra>",
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
        title="Calendrier de présence — heures pointées par technicien",
        xaxis_title="Date", yaxis_title="Technicien",
        height=max(400, 30 * len(techniciens) + 150),
        margin=dict(l=200, r=20, t=60, b=60),
        plot_bgcolor="white",
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
    )
    st.plotly_chart(fig, use_container_width=True)

    col_leg = st.columns(len(STATUS_LABELS))
    for i, (status, label) in enumerate(STATUS_LABELS.items()):
        with col_leg[i]:
            color = STATUS_COLORS[status]
            st.markdown(
                f'<div style="background:{color};border-radius:4px;padding:4px 8px;'
                f'text-align:center;font-size:12px;color:{"black" if status != PresenceStatus.ABSENT else "white"}">'
                f'{label}</div>',
                unsafe_allow_html=True,
            )


def _render_monthly_summary_chart(df_filtered: pd.DataFrame, daily_stats: pd.DataFrame) -> None:
    if daily_stats.empty or "date" not in daily_stats.columns:
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

def _render_export_section(df_presence: pd.DataFrame, builder: "ExhaustiviteBuilder") -> None:
    st.markdown("### 📥 Export Suivi de Présence")
    st.caption("Génère un fichier Excel avec une feuille par équipe.")

    col_e1, col_e2 = st.columns([2, 2])

    with col_e1:
        equipes_dispo  = builder.get_equipes_list(df_presence)
        equipes_export = st.multiselect(
            "Équipes à inclure dans l'export",
            options=equipes_dispo,
            default=equipes_dispo,
            key="export_equipes",
        )

    with col_e2:
        mois_dispo      = builder.get_mois_list(df_presence)
        trimestres      = _get_trimestres(df_presence)
        periode_options = ["Toute la période"] + trimestres + mois_dispo
        periode_export  = st.selectbox(
            "Période",
            options=periode_options,
            index=0,
            key="export_periode",
            help="Sélectionner un mois, un trimestre ou toute la période",
        )

    if not equipes_export:
        st.warning("Sélectionnez au moins une équipe.")
        return

    df_prev  = _filter_for_export(df_presence, equipes_export, periode_export)
    nb_tech  = df_prev["salarie_nom"].nunique() if "salarie_nom" in df_prev.columns else 0
    nb_jours = df_prev["date"].nunique() if "date" in df_prev.columns else 0

    st.info(
        f"**Périmètre export** : {len(equipes_export)} équipe(s) · "
        f"{nb_tech} technicien(s) · {nb_jours} jour(s) · Période : {periode_export}"
    )

    if st.button("⬇️ Générer et télécharger l'Excel", type="primary", key="export_btn"):
        with st.spinner("Génération en cours…"):
            try:
                excel_bytes = _build_export_excel_v2(df_presence, equipes_export, periode_export)
                nom_fichier = _nom_fichier_export(equipes_export, periode_export)
                st.download_button(
                    label=f"📄 Télécharger {nom_fichier}",
                    data=excel_bytes,
                    file_name=nom_fichier,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="export_download",
                )
                st.success(f"✅ Fichier prêt — {len(equipes_export)} feuille(s) générée(s)")
            except Exception as exc:
                st.error(f"Erreur lors de la génération : {exc}")


def _get_trimestres(df_presence: pd.DataFrame) -> list[str]:
    if "date" not in df_presence.columns:
        return []
    dates      = pd.to_datetime(df_presence["date"], errors="coerce").dropna()
    trimestres = set()
    for d in dates:
        q = (d.month - 1) // 3 + 1
        trimestres.add(f"T{q} {d.year}")
    return sorted(trimestres)


def _filter_for_export(df_presence: pd.DataFrame, equipes: list[str], periode: str) -> pd.DataFrame:
    df = df_presence.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if equipes and "equipe_nom" in df.columns:
        df = df[df["equipe_nom"].isin(equipes)]

    if periode != "Toute la période":
        if periode.startswith("T"):
            parts      = periode.split()
            trimestre  = int(parts[0][1])
            annee      = int(parts[1])
            mois_debut = (trimestre - 1) * 3 + 1
            mois_fin   = trimestre * 3
            df = df[
                (df["date"].dt.year  == annee) &
                (df["date"].dt.month >= mois_debut) &
                (df["date"].dt.month <= mois_fin)
            ]
        else:
            df["_mois"] = df["date"].dt.to_period("M").astype(str)
            df = df[df["_mois"] == periode].drop(columns=["_mois"], errors="ignore")

    return df


def _build_export_excel_v2(df_presence: pd.DataFrame, equipes: list[str], periode: str) -> bytes:
    """
    Génère un Excel multi-feuilles — une feuille par équipe.
    FIX : utilise les valeurs du pivot directement (pas cell.value)
    pour éviter les problèmes de lecture openpyxl après to_excel().
    """
    from io import BytesIO
    from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    df = _filter_for_export(df_presence, equipes, periode)
    if df.empty:
        raise ValueError("Aucune donnée pour cette sélection.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    output = BytesIO()

    VERT  = "C6EFCE"
    ROUGE = "FFC7CE"
    GRIS  = "D9D9D9"
    BLEU  = "DDEBF7"
    NAVY  = "002060"
    JAUNE = "FFF2CC"

    thin   = Side(style="thin", color="CCCCCC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    _JOURS_FR = {0: "Lun", 1: "Mar", 2: "Mer", 3: "Jeu", 4: "Ven", 5: "Sam", 6: "Dim"}

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for equipe in sorted(equipes):
            df_eq = df[df["equipe_nom"] == equipe].copy() if "equipe_nom" in df.columns else df.copy()
            if df_eq.empty:
                continue

            hr_col = "h_totale" if "h_totale" in df_eq.columns else "hr_totale"
            if hr_col not in df_eq.columns:
                df_eq[hr_col] = 0.0

            df_eq["date"]   = pd.to_datetime(df_eq["date"]).dt.normalize()
            tous_jours      = pd.date_range(start=df_eq["date"].min(), end=df_eq["date"].max(), freq="D")
            tous_jours_norm = pd.DatetimeIndex([d.normalize() for d in tous_jours])

            # ── Pivot binaire (0/1) ───────────────────────────────────
            pivot = df_eq.pivot_table(
                index="salarie_nom", columns="date", values=hr_col, aggfunc="sum",
            ).reindex(columns=tous_jours_norm, fill_value=0).fillna(0)

            date_cols               = [c for c in pivot.columns if isinstance(c, pd.Timestamp)]
            pivot_bin               = (pivot[date_cols] > 0).astype(int)  # 0/1
            pivot_bin.index.name    = "Technicien"

            nb_jours_ouvres         = sum(1 for d in tous_jours_norm if d.weekday() < 5)
            jours_presents          = pivot_bin.sum(axis=1)

            # ── Écriture Excel sans contenu (structure vide) ──────────
            # On écrit un DataFrame vide avec juste les index/colonnes
            # pour créer la feuille, puis on remplit manuellement
            df_empty = pd.DataFrame(
                index=pivot_bin.index,
                columns=[f"{j.strftime('%d/%m')}" for j in tous_jours] +
                        ["Jours présents", "Jours ouvrés", "Taux présence"]
            )
            df_empty.index.name = "Technicien"
            sheet_name = equipe[:31]
            df_empty.to_excel(writer, sheet_name=sheet_name, startrow=2)

            ws = writer.sheets[sheet_name]

            # ── Titre ─────────────────────────────────────────────────
            ws["A1"] = f"Suivi Présence — {equipe} — {periode}"
            ws["A1"].font      = Font(bold=True, color="FFFFFF", size=12)
            ws["A1"].fill      = PatternFill("solid", fgColor=NAVY)
            ws["A1"].alignment = Alignment(horizontal="left")

            nb_cols_date   = len(tous_jours)
            techniciens    = list(pivot_bin.index)
            nb_techniciens = len(techniciens)

            # ── En-têtes dates (ligne 3) ──────────────────────────────
            for col_idx, jour in enumerate(tous_jours, start=2):
                col_letter        = get_column_letter(col_idx)
                cell_h            = ws.cell(row=3, column=col_idx)
                cell_h.value      = f"{jour.strftime('%d/%m')}\n{_JOURS_FR[jour.weekday()]}"
                cell_h.alignment  = Alignment(wrap_text=True, horizontal="center", vertical="center")
                cell_h.fill       = PatternFill("solid", fgColor=BLEU)
                cell_h.font       = Font(bold=True, size=8)
                cell_h.border     = border
                ws.column_dimensions[col_letter].width = 7

            # ── Données (lignes 4+) en lisant pivot_bin directement ───
            for t_idx, tech in enumerate(techniciens):
                row_idx = 4 + t_idx

                for col_idx, jour in enumerate(tous_jours_norm, start=2):
                    cell       = ws.cell(row=row_idx, column=col_idx)
                    is_weekend = jour.weekday() >= 5

                    if is_weekend:
                        cell.fill  = PatternFill("solid", fgColor=GRIS)
                        cell.value = ""
                    else:
                        # ← lecture directe du pivot, pas de cell.value
                        est_present = int(pivot_bin.loc[tech, jour]) == 1 if jour in pivot_bin.columns else False
                        if est_present:
                            cell.fill  = PatternFill("solid", fgColor=VERT)
                            cell.value = "P"
                            cell.font  = Font(bold=True, size=9, color="276221")
                        else:
                            cell.fill  = PatternFill("solid", fgColor=ROUGE)
                            cell.value = ""

                    cell.alignment = Alignment(horizontal="center")
                    cell.border    = border

                # ── Colonnes résumé ───────────────────────────────────
                jp   = int(jours_presents.loc[tech])
                taux = round(jp / nb_jours_ouvres, 3) if nb_jours_ouvres > 0 else 0

                for i, (col_name, val) in enumerate([
                    ("Jours présents", jp),
                    ("Jours ouvrés",   nb_jours_ouvres),
                    ("Taux présence",  taux),
                ]):
                    col_idx    = nb_cols_date + 2 + i
                    col_letter = get_column_letter(col_idx)
                    cell       = ws.cell(row=row_idx, column=col_idx)
                    cell.value = val
                    cell.alignment = Alignment(horizontal="center")
                    cell.border    = border

                    if col_name == "Taux présence":
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

            # ── En-têtes colonnes résumé (ligne 3) ────────────────────
            for i, col_name in enumerate(["Jours présents", "Jours ouvrés", "Taux présence"]):
                col_idx       = nb_cols_date + 2 + i
                col_letter    = get_column_letter(col_idx)
                cell_h        = ws.cell(row=3, column=col_idx)
                cell_h.value  = col_name
                cell_h.fill   = PatternFill("solid", fgColor="FFC000")
                cell_h.font   = Font(bold=True, size=9)
                cell_h.alignment = Alignment(horizontal="center", wrap_text=True)
                cell_h.border = border
                ws.column_dimensions[col_letter].width = 12

            # ── Colonne technicien ────────────────────────────────────
            ws.column_dimensions["A"].width  = 28
            ws.cell(row=3, column=1).value   = "Technicien"
            ws.cell(row=3, column=1).font    = Font(bold=True, color="FFFFFF", size=10)
            ws.cell(row=3, column=1).fill    = PatternFill("solid", fgColor=NAVY)
            ws.cell(row=3, column=1).alignment = Alignment(horizontal="left")
            ws.cell(row=3, column=1).border  = border

            for t_idx in range(nb_techniciens):
                cell           = ws.cell(row=4 + t_idx, column=1)
                cell.font      = Font(bold=True, size=9)
                cell.fill      = PatternFill("solid", fgColor="EEF3FA")
                cell.alignment = Alignment(horizontal="left", vertical="center")
                cell.border    = border

            # ── Hauteurs lignes ───────────────────────────────────────
            ws.row_dimensions[1].height = 22
            ws.row_dimensions[3].height = 36
            for t_idx in range(nb_techniciens):
                ws.row_dimensions[4 + t_idx].height = 18

            ws.freeze_panes = "B4"

    output.seek(0)
    return output.read()


def _nom_fichier_export(equipes: list[str], periode: str) -> str:
    periode_clean = periode.replace(" ", "_").replace("/", "-")
    if len(equipes) == 1:
        equipe_clean = equipes[0][:20].replace(" ", "_")
        return f"Presence_{equipe_clean}_{periode_clean}.xlsx"
    return f"Presence_Toutes_Equipes_{periode_clean}.xlsx"
