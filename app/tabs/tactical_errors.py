"""
tactical_errors.py — Aba 4: Detecção de Erros Táticos via Model-Driven Random Forest.
"""

import streamlit as st

from app.services import (
    get_available_replays,
    get_detector,
    load_feature_dataset,
)


def tab_errors() -> None:
    """Aba 4: Diagnóstico detalhado de erros de posicionamento e telemetria por replay."""
    st.header("Erros Táticos")
    st.markdown(
        "Diagnóstico alimentado pelo **Random Forest** com validação cruzada. "
        "Os erros são ponderados pela **importância que o modelo atribui a cada feature** "
        "na hora de diferenciar ranks altos de baixos."
    )

    dataset = load_feature_dataset()
    if dataset is None:
        st.error("Dataset de features não encontrado. Execute a extração de dados primeiro.")
        return

    df, X, y, feature_names, _ = dataset

    df_hash = f"{len(df)}_{len(feature_names)}"
    detector, predictions = get_detector(df_hash, X, y, feature_names)

    if not detector.is_built:
        st.warning("Perfil de referência não pôde ser construído (sem amostras 'higher').")
        return

    available = get_available_replays()
    replay_name_to_idx = {}
    counter = 0
    for lbl, replays in available.items():
        for r in replays:
            replay_name_to_idx[r["name"]] = counter
            counter += 1

    total_errors = 0
    for i, (idx, row) in enumerate(df.iterrows()):
        replay_short = row["replay_name"][:25] + "..."
        team_label = row.get("team_label", "?")
        rank_badge = "🔵 Higher" if row["label"] == "higher" else "🟠 Lower"

        pred = predictions[i] if i < len(predictions) else "?"
        pred_icon = "✅" if pred == row["label"] else "❌"

        features_dict = {col: row[col] for col in feature_names}
        errors = detector.detect_errors(features_dict)
        total_errors += len(errors)

        icon = "🔴" if any(e["severity"] == "alta" for e in errors) else "🟡" if errors else "🟢"
        with st.expander(
            f"{icon} [{rank_badge}] {replay_short} — Time {team_label} "
            f"({len(errors)} erros) — Pred: {pred} {pred_icon}"
        ):
            if row["replay_name"] in replay_name_to_idx:
                if st.button("▶ Assistir Replay", key=f"play_{idx}"):
                    st.session_state.pending_replay = replay_name_to_idx[row["replay_name"]]
                    st.rerun()

            if not errors:
                st.success("Nenhum erro tático detectado para este time!")
            else:
                for err in errors:
                    sev_class = "error-alta" if err["severity"] == "alta" else "error-media"
                    sev_badge = "🔴 ALTA" if err["severity"] == "alta" else "🟡 MÉDIA"
                    st.markdown(
                        f'<div class="error-card {sev_class}">'
                        f'<strong>{sev_badge}</strong> · <strong>{err.get("type", "Erro")}</strong><br>'
                        f'{err["description"]}<br>'
                        f'<span class="uu-info">'
                        f'Valor medido: {err["value"]} · Média de referência: {err.get("reference", err.get("reference_mean", 0))} '
                        f'(desvio z: {err.get("z_score", 0):+.1f}σ) · Importância: {err.get("importance", 0):.1%}'
                        f'</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
