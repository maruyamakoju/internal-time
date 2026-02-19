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

**pa-F1 0.814** overall on the NAB benchmark (52 real-world time series files):

| Category | Files | pa-F1 |
|---|---|---|
| artificialWithAnomaly | 6 | **0.947** |
| realTweets | 10 | **0.887** |
| realKnownCause | 7 | **0.844** |
| realAWSCloudwatch | 16 | **0.836** |
| realTraffic | 7 | 0.722 |
| realAdExchange | 6 | 0.574 |
| **Overall** | **52** | **0.814** |

## How It Works

A GRU processes the time series step by step:
1. A **self-model** predicts the next hidden state
2. The **prediction error** measures temporal surprise
3. An **internal time head** outputs Δτ — the agent's internal clock speed
4. High error + unusual Δτ → anomaly signal

MIT License
