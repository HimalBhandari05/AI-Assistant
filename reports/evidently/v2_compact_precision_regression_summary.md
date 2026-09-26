# Evidently AI Regression Analysis Report: `v2_compact_precision` vs `v1_baseline`

**Overall Status**: ✅ NO REGRESSION DETECTED
**Timestamp**: `2026-09-26T13:35:25.681242+00:00`

## 1. Regression Metrics Overview

| Metric | Reference (`v1_baseline`) | Current (`v2_compact_precision`) | Status | Threshold |
| :--- | :---: | :---: | :---: | :--- |
| **Benchmark Pass Rate** | `100.0%` | `100.0%` | ✅ | $\ge 95.0\%$ |
| **Task Completion Rate (TCR)** | `100.0%` | `100.0%` | ✅ | $\ge 95.0\%$ |
| **Tool-Call Correctness (TCC)** | `100.0%` | `100.0%` | ✅ | $\ge 95.0\%$ |
| **Average Trajectory Length** | `2.12` | `1.94` | ✅ | $\le 3.18$ |
| **Average Tokens per Query** | `153.5` | `105.8` | ✅ | $\le 268.6$ |
| **Trajectory Efficiency Score (TES)** | `47.2` | `51.5` | ✅ | $\ge 35.0$ |
| **Token Cost Ratio** | `1.000` | `0.689` | ✅ | $\le 1.75$ |

## 2. Regression Verification

All evaluation metrics and custom agentic performance scores remained within configured regression thresholds.

## 3. Benchmark Correctness Analysis

All **16/16** fixed benchmark test cases completed successfully with zero functional regressions.
