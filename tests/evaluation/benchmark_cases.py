"""Curated benchmark dataset for evaluating W16 Agentic Assistant behavior."""

from typing import List, Optional
from pydantic import BaseModel, Field


class BenchmarkCase(BaseModel):
    """Structured representation of a standardized evaluation test case."""
    id: str = Field(..., description="Unique case identifier (e.g. 'case_calc_01').")
    category: str = Field(..., description="Category: 'single_step_calc', 'single_step_rag', 'multi_step_retrieval', 'retrieval_and_calculator', 'clarification', 'boundary_stop'.")
    question: str = Field(..., description="The exact user query to evaluate.")
    expected_status: str = Field(default="completed", description="Expected agent status: 'completed' or 'clarification_required'.")
    required_tools: List[str] = Field(default_factory=list, description="Tools that MUST be invoked during the trajectory.")
    forbidden_tools: List[str] = Field(default_factory=list, description="Tools that MUST NOT be invoked.")
    expected_answer_keywords: List[str] = Field(default_factory=list, description="Keywords or phrases expected in the final answer.")
    expected_numerical_result: Optional[float] = Field(default=None, description="Expected exact numerical value for math tasks.")
    max_acceptable_trajectory_length: int = Field(default=5, description="Upper bound on acceptable steps for this task.")
    description: str = Field(default="", description="High-level description of what this case verifies.")


# Curated benchmark dataset of 16 test cases grounded in documents/sample.txt and math operations
BENCHMARK_CASES: List[BenchmarkCase] = [
    # -------------------------------------------------------------------------
    # Category A: Single-Step Tasks (Arithmetic & Baseline)
    # -------------------------------------------------------------------------
    BenchmarkCase(
        id="case_calc_01",
        category="single_step_calc",
        question="What is 45 multiplied by 12?",
        expected_status="completed",
        required_tools=["calculator"],
        forbidden_tools=["rag_search"],
        expected_numerical_result=540.0,
        expected_answer_keywords=["540"],
        max_acceptable_trajectory_length=2,
        description="Verify direct calculation execution without unnecessary document searches.",
    ),
    BenchmarkCase(
        id="case_calc_02",
        category="single_step_calc",
        question="Calculate 256 divided by 16.",
        expected_status="completed",
        required_tools=["calculator"],
        forbidden_tools=["rag_search"],
        expected_numerical_result=16.0,
        expected_answer_keywords=["16"],
        max_acceptable_trajectory_length=2,
        description="Verify arithmetic division with direct calculator dispatch.",
    ),
    BenchmarkCase(
        id="case_calc_03",
        category="single_step_calc",
        question="What is 1450 added to 3250?",
        expected_status="completed",
        required_tools=["calculator"],
        forbidden_tools=["rag_search"],
        expected_numerical_result=4700.0,
        expected_answer_keywords=["4700"],
        max_acceptable_trajectory_length=2,
        description="Verify multi-digit addition.",
    ),

    # -------------------------------------------------------------------------
    # Category B: Single Retrieval Tasks (Direct Knowledge Base Lookup)
    # -------------------------------------------------------------------------
    BenchmarkCase(
        id="case_rag_01",
        category="single_step_rag",
        question="What is Retrieval-Augmented Generation (RAG) according to the documents?",
        expected_status="completed",
        required_tools=["rag_search"],
        forbidden_tools=["calculator"],
        expected_answer_keywords=["retrieves", "knowledge", "hallucinations"],
        max_acceptable_trajectory_length=2,
        description="Verify single-pass retrieval and grounded extraction of RAG architectural definition.",
    ),
    BenchmarkCase(
        id="case_rag_02",
        category="single_step_rag",
        question="What is deep learning and what kind of unstructured data does it process according to the text?",
        expected_status="completed",
        required_tools=["rag_search"],
        forbidden_tools=["calculator"],
        expected_answer_keywords=["neural networks", "text", "images", "audio"],
        max_acceptable_trajectory_length=2,
        description="Verify retrieval of deep learning characteristics and supported data types.",
    ),
    BenchmarkCase(
        id="case_rag_03",
        category="single_step_rag",
        question="What is the time complexity of linear search according to the documents?",
        expected_status="completed",
        required_tools=["rag_search"],
        forbidden_tools=["calculator"],
        expected_answer_keywords=["linear search", "O(n)"],
        max_acceptable_trajectory_length=2,
        description="Verify retrieval of linear search sequential scanning and O(n) complexity.",
    ),

    # -------------------------------------------------------------------------
    # Category C: Multi-Step Retrieval Tasks (Multi-Topic Synthesis & Comparison)
    # -------------------------------------------------------------------------
    BenchmarkCase(
        id="case_multi_01",
        category="multi_step_retrieval",
        question="Compare linear search and binary search time complexities from the documents.",
        expected_status="completed",
        required_tools=["rag_search"],
        expected_answer_keywords=["O(n)", "O(log n)", "binary search", "linear search"],
        max_acceptable_trajectory_length=4,
        description="Verify multi-step retrieval and comparative synthesis of both algorithm complexities.",
    ),
    BenchmarkCase(
        id="case_multi_02",
        category="multi_step_retrieval",
        question="What are the key relationships and differences between Artificial Intelligence, Machine Learning, and Deep Learning according to the documents?",
        expected_status="completed",
        required_tools=["rag_search"],
        expected_answer_keywords=["subset", "data", "neural networks"],
        max_acceptable_trajectory_length=4,
        description="Verify synthesis across the AI hierarchical definitions.",
    ),
    BenchmarkCase(
        id="case_multi_03",
        category="multi_step_retrieval",
        question="How does Retrieval-Augmented Generation improve language models compared to standard AI systems without retraining?",
        expected_status="completed",
        required_tools=["rag_search"],
        expected_answer_keywords=["knowledge bases", "hallucinations", "retraining", "attribution"],
        max_acceptable_trajectory_length=4,
        description="Verify synthesis of RAG advantages and update mechanisms without model retraining.",
    ),

    # -------------------------------------------------------------------------
    # Category D: Retrieval + Calculator (Tool Chaining)
    # -------------------------------------------------------------------------
    BenchmarkCase(
        id="case_chain_01",
        category="retrieval_and_calculator",
        question="According to the documents, if a linear search examines 500 items sequentially and binary search takes 9 comparisons, calculate the difference in operations.",
        expected_status="completed",
        required_tools=["rag_search", "calculator"],
        expected_numerical_result=491.0,
        expected_answer_keywords=["491"],
        max_acceptable_trajectory_length=4,
        description="Verify retrieval of algorithm context followed by subtraction calculation.",
    ),
    BenchmarkCase(
        id="case_chain_02",
        category="retrieval_and_calculator",
        question="If an array of 250 elements is searched with linear search taking 250 operations and we repeat this test 50 times, what is the total number of operations?",
        expected_status="completed",
        required_tools=["calculator"],
        expected_numerical_result=12500.0,
        expected_answer_keywords=["12500"],
        max_acceptable_trajectory_length=4,
        description="Verify tool chaining for multi-iteration search operation count.",
    ),
    BenchmarkCase(
        id="case_chain_03",
        category="retrieval_and_calculator",
        question="If a deep learning system processes 4 layers of neural networks with 8 feature maps per layer, what is the total number of feature maps computed?",
        expected_status="completed",
        required_tools=["calculator"],
        expected_numerical_result=32.0,
        expected_answer_keywords=["32"],
        max_acceptable_trajectory_length=4,
        description="Verify neural network layer calculation chained with problem context.",
    ),

    # -------------------------------------------------------------------------
    # Category E: Clarification (Ambiguous Inputs)
    # -------------------------------------------------------------------------
    BenchmarkCase(
        id="case_clarify_01",
        category="clarification",
        question="What is its exact time complexity?",
        expected_status="clarification_required",
        required_tools=["ask_user_clarification"],
        expected_answer_keywords=["which", "algorithm"],
        max_acceptable_trajectory_length=2,
        description="Verify ambiguous pronoun 'its' prompts clarification instead of hallucinated answer.",
    ),
    BenchmarkCase(
        id="case_clarify_02",
        category="clarification",
        question="How many layers does it have?",
        expected_status="clarification_required",
        required_tools=["ask_user_clarification"],
        expected_answer_keywords=["which", "model", "neural network"],
        max_acceptable_trajectory_length=2,
        description="Verify ambiguous pronoun 'it' triggers clarification without guessing.",
    ),

    # -------------------------------------------------------------------------
    # Category F: Boundary / Quick-Stop Tests
    # -------------------------------------------------------------------------
    BenchmarkCase(
        id="case_bound_01",
        category="boundary_stop",
        question="What is 100 minus 37?",
        expected_status="completed",
        required_tools=["calculator"],
        forbidden_tools=["rag_search"],
        expected_numerical_result=63.0,
        expected_answer_keywords=["63"],
        max_acceptable_trajectory_length=2,
        description="Verify fast termination on elementary arithmetic (1 tool call + final answer).",
    ),
    BenchmarkCase(
        id="case_bound_02",
        category="boundary_stop",
        question="What is the primary benefit of RAG according to the documents?",
        expected_status="completed",
        required_tools=["rag_search"],
        forbidden_tools=["calculator"],
        expected_answer_keywords=["hallucinations", "knowledge"],
        max_acceptable_trajectory_length=2,
        description="Verify fast termination on direct document question (1 retrieval + final answer).",
    ),
]
