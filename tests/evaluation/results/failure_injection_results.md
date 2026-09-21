# W16 Fault Recovery & Failure Injection Report

## 1. Failure Injection Objective

The objective of this assessment is to demonstrate that the W16 agent does not blindly assume tool success or hallucinate answers when intermediate execution fails. Specifically, the experiment verifies that:
1. Failures are captured and presented as structured observations.
2. The agent detects the failure and decides whether recovery is feasible.
3. When recovery is possible (e.g. query reformulation or argument correction), the agent performs corrective actions.
4. When recovery is impossible, the agent terminates safely without fabricating unsupported answers.

## 2. Injection Methodology

* **Controlled Dependency Injection / Interception**: Tool errors and empty retrieval payloads are injected during specific trajectory steps without modifying permanent ChromaDB storage or production tool handlers.
* **Structured Error Contracts**: Errors are delivered via standard `ToolError(type, message)` payloads, matching the Phase 1 tool contract.
* **No Hardcoded Recovery**: The agent loop dynamically observes the failure observation in its scratchpad and independently determines the next action.

## 3. Evaluated Failure Scenarios

| Case ID | Category | Injection Scenario | Recoverable | Expected Classification |
| :--- | :--- | :--- | :---: | :--- |
| `FI-01` | `empty_retrieval` | First rag_search returns 0 chunks. Agent observes empty resu... | Yes | `Soft Failure` |
| `FI-02` | `tool_error` | First calculator call attempts division by zero (256/0). Age... | Yes | `Soft Failure` |
| `FI-03` | `cascading_error` | Initial retrieval returns empty result; second retrieval suc... | Yes | `Cascading Soft Failure` |
| `FI-04` | `iteration_budget` | Continuous search queries repeatedly return 0 evidence. Agen... | No | `Hard Failure` |

## 4. Overall Fault Recovery Results

| Metric | Result |
| :--- | ---: |
| **Total Injected Scenarios** | `4` |
| **Failures Detected** | `4` / `4` |
| **Failure Detection Rate** | **`100.0%`** |
| **Recoverable Scenarios** | `3` |
| **Failures Successfully Recovered** | `3` / `3` |
| **Failure Recovery Rate** | **`100.0%`** |
| **Average Trajectory Length (Fault Recovery)** | `4.00 steps` |
| **Average Tokens per Scenario** | `393.8 tokens` |
| **Total Tokens Consumed** | `1575` tokens |

## 5. Per-Scenario Failure Analysis

| Case ID | Injection Type | Detected | Recovered | Final Status | Steps | Tokens | Classification | Overhead |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :--- | :--- |
| `FI-01` | `empty_first_retrieval` | ✅ Yes | ✅ Yes | `completed` | `3` | `290` | `Soft Failure` | +1 steps / +135 tok |
| `FI-02` | `calculator_division_by_zero` | ✅ Yes | ✅ Yes | `completed` | `3` | `155` | `Soft Failure` | +1 steps / +75 tok |
| `FI-03` | `cascading_retrieval_and_tool_error` | ✅ Yes | ✅ Yes | `completed` | `5` | `510` | `Cascading Soft Failure` | +2 steps / +245 tok |
| `FI-04` | `repeated_empty_retrieval` | ✅ Yes | N/A | `max_iterations_exceeded` | `5` | `620` | `Hard Failure` | N/A |

## 6. Failure Taxonomy Breakdown

| Failure Classification | Measured Count | Percentage | Operational Meaning |
| :--- | ---: | ---: | :--- |
| **Hard Failure** | `1` | `25.0%` | Controlled budget limit exhaustion on unrecoverable query (`FI-04`). |
| **Soft Failure** | `2` | `50.0%` | Single-step error detection and immediate 1-step recovery (`FI-01`, `FI-02`). |
| **Cascading Soft Failure** | `1` | `25.0%` | Compound multi-step recovery requiring multiple corrective iterations (`FI-03`). |

## 7. Cost of Recovery Analysis

Comparing normal baseline executions against injected failure recovery trajectories:

| Scenario | Normal Steps | Fault Steps | Step Delta | Normal Tokens | Fault Tokens | Token Overhead |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `FI-01` (empty_retrieval) | `2` | `3` | **`+1`** | `155` | `290` | **`+135 (87.1%)`** |
| `FI-02` (tool_error) | `2` | `3` | **`+1`** | `80` | `155` | **`+75 (93.8%)`** |
| `FI-03` (cascading_error) | `3` | `5` | **`+2`** | `265` | `510` | **`+245 (92.5%)`** |

### Key Findings:
1. **Query Reformulation Overhead**: When initial search fails, reformulating and re-executing retrieval adds exactly **+1 step** and **~135-155 tokens** of prompt/observation overhead.
2. **Arithmetic Correction Overhead**: Correcting an invalid tool call requires **+1 step** and **~75 tokens**.
3. **Cascading Overhead**: Compounding multi-step errors add **+2 steps** and **~245 tokens** before reaching verified completion.

## 8. Security & Safety Verification

* **Hallucination Prevention**: In all failure scenarios (including empty retrieval and budget exhaustion), the agent **strictly refrained from fabricating unsupported facts**. It never converted an empty observation into a confident answer.
* **Graceful Degradation**: On unrecoverable queries (`FI-04`), the agent terminated with `max_iterations_exceeded` and provided a transparent explanation rather than crashing or looping indefinitely.
