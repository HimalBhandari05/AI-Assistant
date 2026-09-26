# Evidently AI Regression Analysis Report: `v1_baseline` vs `v1_baseline`

**Overall Status**: ✅ NO REGRESSION DETECTED
**Timestamp**: `2026-09-26T13:35:23.099387+00:00`

## 1. Regression Metrics Overview

| Metric | Reference (`v1_baseline`) | Current (`v1_baseline`) | Status | Threshold |
| :--- | :---: | :---: | :---: | :--- |
| **Benchmark Pass Rate** | `100.0%` | `100.0%` | ✅ | $\ge 95.0\%$ |
| **Task Completion Rate (TCR)** | `100.0%` | `100.0%` | ✅ | $\ge 95.0\%$ |
| **Tool-Call Correctness (TCC)** | `100.0%` | `100.0%` | ✅ | $\ge 95.0\%$ |
| **Average Trajectory Length** | `2.12` | `2.12` | ✅ | $\le 3.18$ |
| **Average Tokens per Query** | `153.5` | `153.5` | ✅ | $\le 268.6$ |
| **Trajectory Efficiency Score (TES)** | `47.2` | `47.2` | ✅ | $\ge 35.0$ |
| **Token Cost Ratio** | `1.000` | `1.000` | ✅ | $\le 1.75$ |

## 2. Regression Verification

All evaluation metrics and custom agentic performance scores remained within configured regression thresholds.

## 3. Benchmark Correctness Analysis

All **16/16** fixed benchmark test cases completed successfully with zero functional regressions.
