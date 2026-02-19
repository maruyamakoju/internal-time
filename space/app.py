"""
Gradio demo for internal-time anomaly detector (Japanese UI).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gradio as gr
from pathlib import Path

from internal_time import TemporalAnomalyDetector
from internal_time.viz import plot_anomaly_scores, plot_detail


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def _base_signal(T: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, T)
    x = np.sin(t) + 0.5 * np.sin(2.3 * t) + 0.15 * rng.standard_normal(T)
    return x.astype(np.float32)


ANOMALY_TYPES = {
    "スパイク（突発的な外れ値）": "Spike",
    "レベルシフト（平均値の変化）": "Step (mean shift)",
    "振動変化（周波数の変化）": "Oscillation (freq change)",
    "分散バースト（ノイズ急増）": "Variance burst",
    "複合異常（スパイク＋レベルシフト）": "Mixed",
}


def make_synthetic(anomaly_label: str, T: int = 600, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    anomaly_type = ANOMALY_TYPES.get(anomaly_label, anomaly_label)
    rng = np.random.default_rng(seed)
    x = _base_signal(T, seed)
    labels = np.zeros(T, dtype=np.int32)

    if anomaly_type == "Spike":
        positions = rng.integers(T // 2, T, size=3)
        for p in positions:
            x[p] += rng.choice([-1, 1]) * rng.uniform(3, 5)
            labels[max(0, p - 2) : p + 3] = 1

    elif anomaly_type == "Step (mean shift)":
        start, end = T // 2, T // 2 + 80
        x[start:end] += 2.5
        labels[start:end] = 1

    elif anomaly_type == "Oscillation (freq change)":
        start, end = T // 2, T // 2 + 100
        t2 = np.linspace(0, 12 * np.pi, end - start)
        x[start:end] += 1.2 * np.sin(t2)
        labels[start:end] = 1

    elif anomaly_type == "Variance burst":
        start, end = T // 2, T // 2 + 80
        x[start:end] += rng.standard_normal(end - start) * 1.8
        labels[start:end] = 1

    elif anomaly_type == "Mixed":
        p = T // 2 + 50
        x[p] += 4.0
        labels[max(0, p - 2) : p + 3] = 1
        s, e = T // 2 + 150, T // 2 + 230
        x[s:e] += 2.2
        labels[s:e] = 1

    return x.reshape(-1, 1), labels


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def run_detection(
    values: np.ndarray,
    labels: np.ndarray | None,
    epochs: int,
    threshold: float,
    train_ratio: float = 0.5,
) -> tuple[plt.Figure, plt.Figure, str]:
    T = len(values)
    split = int(T * train_ratio)
    train_data = values[:split]

    det = TemporalAnomalyDetector(
        input_dim=values.shape[1],
        hidden_dim=48,
        epochs=epochs,
        window_size=min(50, max(10, split // 5)),
        patience=12,
        lr=3e-4,
    )
    det.fit(train_data, verbose=False)

    scores = det.score(values)
    detail = det.score_detail(values)
    preds = (scores > threshold).astype(np.int32)

    fig1 = plot_anomaly_scores(
        data=values,
        scores=scores,
        threshold=threshold,
        labels_true=labels,
        title="異常検知結果 — internal-time",
        figsize=(12, 5),
    )

    fig2 = plot_detail(
        detail=detail,
        data=values,
        title="内部シグナル (\u0394\u03c4, pred_error, \u03b1)",
        figsize=(12, 5),
    )

    lines = []
    if labels is not None and labels.sum() > 0:
        from internal_time.benchmark import compute_metrics
        m = compute_metrics(preds, labels)
        lines += [
            f"**点単位 F1**: {m['f1']:.3f}　(適合率 {m['precision']:.3f}、再現率 {m['recall']:.3f})",
            f"**点調整 F1 (pa-F1)**: {m['pa_f1']:.3f}　(適合率 {m['pa_precision']:.3f}、再現率 {m['pa_recall']:.3f})",
        ]
    lines += [
        f"**スコア範囲**: [{scores.min():.2f}, {scores.max():.2f}]",
        f"**異常フラグ**: {int(preds.sum())} / {T} タイムステップ",
    ]
    stats = "\n\n".join(lines)
    return fig1, fig2, stats


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------

def demo_callback(anomaly_type: str, epochs: int, threshold: float) -> tuple:
    try:
        values, labels = make_synthetic(anomaly_type)
        fig1, fig2, stats = run_detection(
            values, labels, epochs=int(epochs), threshold=float(threshold)
        )
        return fig1, fig2, stats
    except Exception as e:
        return None, None, f"エラーが発生しました: {e}"


def upload_callback(file, epochs: int, threshold: float, label_col: str) -> tuple:
    if file is None:
        return None, None, "CSVファイルをアップロードしてください。"

    try:
        path = file.name if hasattr(file, "name") else file
        df = pd.read_csv(path)
    except Exception as e:
        return None, None, f"CSV読み込みエラー: {e}"

    label_col = (label_col or "").strip()
    labels = None
    skip = {"timestamp"}
    if label_col and label_col in df.columns:
        labels = df[label_col].values.astype(np.int32)
        skip.add(label_col)

    value_cols = [c for c in df.columns if c not in skip]
    if not value_cols:
        return None, None, "数値列が見つかりませんでした。"

    values = df[value_cols].values.astype(np.float32)

    try:
        fig1, fig2, stats = run_detection(
            values, labels, epochs=int(epochs), threshold=float(threshold)
        )
    except Exception as e:
        return None, None, f"検知エラー: {e}"

    return fig1, fig2, stats


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

DESCRIPTION = """
# Internal Time — 時系列異常検知

AIエージェントが自分自身の**内部時計**を学習します。予期しない変化が起きると、内部時計が反応し、
*時間的な驚き*を捉えた自然な異常シグナルを生成します。

- **NABベンチマーク**（52の実世界ファイル）で高精度達成
- 1変量・多変量の時系列データに対応
- ラベル不要 — 完全教師なし学習
"""

REAL_EXAMPLES = {
    "NYC タクシー乗車数（ハロウィン異常）": "examples/nyc_taxi_sample.csv",
}


def real_example_callback(example_name: str, epochs: int, threshold: float) -> tuple:
    path = REAL_EXAMPLES.get(example_name)
    if path is None or not Path(path).exists():
        return None, None, f"サンプルファイルが見つかりません: {path}"
    try:
        df = pd.read_csv(path)
        value_cols = [c for c in df.columns if c != "timestamp"]
        values = df[value_cols].values.astype(np.float32)
        fig1, fig2, stats = run_detection(values, None, epochs=int(epochs), threshold=float(threshold))
        return fig1, fig2, stats
    except Exception as e:
        return None, None, f"エラー: {e}"

with gr.Blocks(theme=gr.themes.Soft(), title="Internal Time 異常検知") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tabs():

        # ---- Tab 1: クイックデモ ----------------------------------------
        with gr.Tab("クイックデモ"):
            gr.Markdown(
                "異常タイプを選んで **実行** を押してください。"
                "前半50%を正常データとして学習し、全体をスコアリングします。"
            )
            with gr.Row():
                atype = gr.Dropdown(
                    list(ANOMALY_TYPES.keys()),
                    value="スパイク（突発的な外れ値）",
                    label="異常タイプ",
                )
                epochs_s = gr.Slider(10, 100, value=40, step=5, label="学習エポック数")
                thr_s = gr.Slider(0.5, 4.0, value=2.0, step=0.1, label="検出閾値 (\u03c3)")
                run_btn = gr.Button("実行", variant="primary")

            stats_out = gr.Markdown(value="← **実行** を押すと結果が表示されます")
            fig_main = gr.Plot(label="時系列データ＋異常スコア")
            fig_detail = gr.Plot(label="内部シグナル（\u0394\u03c4・予測誤差・\u03b1）")

            run_btn.click(
                fn=demo_callback,
                inputs=[atype, epochs_s, thr_s],
                outputs=[fig_main, fig_detail, stats_out],
            )

        # ---- Tab 2: CSVアップロード ----------------------------------------
        with gr.Tab("CSV アップロード"):
            gr.Markdown(
                "数値列を含むCSVファイルをアップロードしてください。\n"
                "- `timestamp` 列は自動的に無視されます\n"
                "- 0/1 のラベル列がある場合は列名を指定するとF1スコアが表示されます"
            )
            with gr.Row():
                csv_file = gr.File(label="CSV ファイル", file_types=[".csv"])
                label_col_in = gr.Textbox(
                    label="ラベル列名（任意）", placeholder="例: label"
                )
            with gr.Row():
                epochs_u = gr.Slider(10, 100, value=40, step=5, label="学習エポック数")
                thr_u = gr.Slider(0.5, 4.0, value=2.0, step=0.1, label="検出閾値 (\u03c3)")
                run_btn_u = gr.Button("実行", variant="primary")

            stats_out_u = gr.Markdown(value="← **実行** を押すと結果が表示されます")
            fig_main_u = gr.Plot(label="時系列データ＋異常スコア")
            fig_detail_u = gr.Plot(label="内部シグナル")

            run_btn_u.click(
                fn=upload_callback,
                inputs=[csv_file, epochs_u, thr_u, label_col_in],
                outputs=[fig_main_u, fig_detail_u, stats_out_u],
            )

        # ---- Tab 3: 実データサンプル ----------------------------------------
        with gr.Tab("実データサンプル"):
            gr.Markdown(
                "実際の時系列データ（NABベンチマーク収録）で試せます。\n"
                "- **NYC タクシー乗車数**: ハロウィン（10月31日）前後の異常な乗車数増加を検知します"
            )
            with gr.Row():
                example_sel = gr.Dropdown(
                    list(REAL_EXAMPLES.keys()),
                    value=list(REAL_EXAMPLES.keys())[0],
                    label="データセット",
                )
                epochs_r = gr.Slider(10, 100, value=30, step=5, label="学習エポック数")
                thr_r = gr.Slider(0.5, 4.0, value=2.0, step=0.1, label="検出閾値 (σ)")
                run_btn_r = gr.Button("実行", variant="primary")

            stats_out_r = gr.Markdown(value="← **実行** を押すと結果が表示されます")
            fig_main_r = gr.Plot(label="時系列データ＋異常スコア")
            fig_detail_r = gr.Plot(label="内部シグナル")

            run_btn_r.click(
                fn=real_example_callback,
                inputs=[example_sel, epochs_r, thr_r],
                outputs=[fig_main_r, fig_detail_r, stats_out_r],
            )

    gr.Markdown(
        "---\n"
        "**仕組み**: GRUが時系列をステップごとに処理します。"
        "*自己モデル*が次の隠れ状態を予測し、予測誤差が高いとき、"
        "エージェントの内部時計 Δτ が反応して*時間的な驚き*を検出します。"
        "正常データのみで学習するため、ラベルは不要です。\n\n"
        "MIT License"
    )


if __name__ == "__main__":
    demo.launch(show_error=True)
