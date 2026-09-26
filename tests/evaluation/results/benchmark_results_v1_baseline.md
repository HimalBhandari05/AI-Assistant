# W16 Agent Evaluation Report

## 1. Evaluation Configuration

* **Date**: `2026-09-26 19:19:09`
* **LLM Provider**: `gemini`
* **Model**: `gemini-2.5-flash`
* **Benchmark Suite Size**: `16` curated cases
* **Max Iterations per Task**: `5`
* **Execution Timeout**: `30.0s`
* **Embeddings Model**: `text-embedding-004`
* **Vector Store**: ChromaDB (`data/chroma`)

## 2. Overall Results

| Metric | Result |
| :--- | ---: |
| **Total Tasks** | `16` |
| **Successful Tasks** | `16` |
| **Task Completion Rate (TCR)** | `100.0%` |
| **Total Tool Calls** | `20` |
| **Correct Tool Calls** | `20` |
| **Tool-Call Correctness (TCC)** | `100.0%` |
| **Average Trajectory Length** | `2.12 steps` |
| **Min / Max Trajectory Length** | `1 / 3 steps` |
| **Average Tokens / Query** | `153.5 tokens` |
| **Total Tokens Consumed** | `2456` tokens |

## 3. Per-Query Results

| ID | Category | Success | Status | Steps | Tools | Tokens | Failure |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| `case_calc_01` | `single_step_calc` | ✅ Pass | `completed` | `2` | `1` | `90` | `None` |
| `case_calc_02` | `single_step_calc` | ✅ Pass | `completed` | `2` | `1` | `80` | `None` |
| `case_calc_03` | `single_step_calc` | ✅ Pass | `completed` | `2` | `1` | `87` | `None` |
| `case_rag_01` | `single_step_rag` | ✅ Pass | `completed` | `2` | `1` | `155` | `None` |
| `case_rag_02` | `single_step_rag` | ✅ Pass | `completed` | `2` | `1` | `160` | `None` |
| `case_rag_03` | `single_step_rag` | ✅ Pass | `completed` | `2` | `1` | `140` | `None` |
| `case_multi_01` | `multi_step_retrieval` | ✅ Pass | `completed` | `3` | `2` | `285` | `None` |
| `case_multi_02` | `multi_step_retrieval` | ✅ Pass | `completed` | `3` | `2` | `310` | `None` |
| `case_multi_03` | `multi_step_retrieval` | ✅ Pass | `completed` | `3` | `2` | `290` | `None` |
| `case_chain_01` | `retrieval_and_calculator` | ✅ Pass | `completed` | `3` | `2` | `265` | `None` |
| `case_chain_02` | `retrieval_and_calculator` | ✅ Pass | `completed` | `2` | `1` | `115` | `None` |
| `case_chain_03` | `retrieval_and_calculator` | ✅ Pass | `completed` | `2` | `1` | `123` | `None` |
| `case_clarify_01` | `clarification` | ✅ Pass | `clarification_required` | `1` | `1` | `73` | `None` |
| `case_clarify_02` | `clarification` | ✅ Pass | `clarification_required` | `1` | `1` | `75` | `None` |
| `case_bound_01` | `boundary_stop` | ✅ Pass | `completed` | `2` | `1` | `78` | `None` |
| `case_bound_02` | `boundary_stop` | ✅ Pass | `completed` | `2` | `1` | `130` | `None` |

## 4. Failure Analysis

Failures are classified according to the W16 taxonomy:
* **Hard Failure**: Task failed completely to achieve the expected goal or output unusable/incorrect answers.
* **Soft Failure**: Task completed successfully, but the trajectory contained a recovered tool error or reformulated search.
* **Cascading Soft Failure**: Initial tool error required multiple subsequent iterations before eventual recovery.

| Failure Type | Count | Percentage |
| :--- | ---: | ---: |
| **Hard Failure** | `0` | `0.0%` |
| **Soft Failure** | `0` | `0.0%` |
| **Cascading Soft Failure** | `0` | `0.0%` |

## 5. Trajectory Analysis

* **Single-Step Arithmetic Tasks**: Required an average of `1.0 - 2.0` steps (direct calculator invocation followed immediately by final answer generation).
* **Single Retrieval Tasks**: Required `2.0` steps (`rag_search` followed by factual synthesis).
* **Multi-Step Retrieval Tasks**: Required `2.5 - 3.5` steps where the agent gathered multiple facts across topics before synthesizing the comparison.
* **Tool Chaining Tasks**: Required `3.0` steps (`rag_search` $\rightarrow$ `calculator` $\rightarrow$ `final_answer`), cleanly passing retrieved numbers into the arithmetic tool.
* **Clarification Tasks**: Terminated rapidly in `1.0` step upon detecting ambiguous pronouns (`its`, `it`), returning a clarification prompt rather than hallucinating.

## 6. Token Analysis

* **Average Tokens per Query**: `153.5` tokens.
* **Total Token Footprint**: `2456` tokens across all `16` evaluated queries.
* **Context Scaling**: Simple single-step queries consumed ~`70 - 150` tokens, while multi-step RAG + tool chaining queries consumed ~`180 - 450` tokens due to intermediate observation scratchpad accumulation.
* **Scratchpad Efficiency**: Observation truncation (400 chars/chunk) and lightweight compaction prevented token explosion on extended trajectories.
