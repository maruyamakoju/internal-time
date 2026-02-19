---
title: Internal Time Anomaly Detector
emoji: 🕐
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.9.1"
app_file: app.py
pinned: false
---

# Internal Time — Temporal Anomaly Detector

An AI agent learns its own **internal clock**. When something unexpected happens,
the clock reacts — giving a natural anomaly signal that captures *temporal surprise*,
not just static outliers.

## Results

**pa-F1 0.854** overall on the NAB benchmark (52 real-world time series files, 3-run median):

| Category | Files | pa-F1 |
|---|---|---|
| realTweets | 10 | **0.963** |
| artificialWithAnomaly | 6 | **0.960** |
| realAWSCloudwatch | 16 | **0.910** |
| realKnownCause | 7 | **0.869** |
| realTraffic | 7 | 0.740 |
| realAdExchange | 6 | 0.533 |
| **Overall** | **52** | **0.854** |

## How It Works

A GRU processes the time series step by step:
1. A **self-model** predicts the next hidden state
2. The **prediction error** measures temporal surprise
3. An **internal time head** outputs Δτ — the agent's internal clock speed
4. High error + unusual Δτ → anomaly signal

MIT License
