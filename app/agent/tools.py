"""Provider-independent Tool Registry and Tool Implementations for Agentic Assistant."""

import inspect
import logging
import time
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

from app.rag.retrieve import retrieve_relevant_chunks
from app.tools import calculator as base_calculator

logger = logging.getLogger("ai_assistant.agent.tools")


class ToolError(BaseModel):
    """Structured error container for tool failures."""
    type: str = Field(..., description="The error type classification (e.g. ValueError, RetrievalError).")
    message: str = Field(..., description="Human-readable description of the error.")


class ToolResult(BaseModel):
    """Standardized result contract returned by all registered tools."""
    success: bool = Field(..., description="True if the tool executed without error, False otherwise.")
    tool: str = Field(..., description="Name of the tool executed.")
    result: Optional[Any] = Field(default=None, description="The payload produced upon successful tool execution.")
    error: Optional[ToolError] = Field(default=None, description="Structured error information if tool execution failed.")
    execution_time_seconds: float = Field(default=0.0, description="Elapsed execution time in seconds.")


def rag_search_tool(query: str, top_k: int = 3) -> dict:
    """Dynamically search the ChromaDB vector knowledge base for relevant document chunks.

    Args:
        query: Search query or keywords to retrieve relevant chunks for.
        top_k: Number of most similar document chunks to retrieve (default: 3).

    Returns:
        A dictionary containing the query, retrieved chunks with metadata, and result count.
    """
    clean_query = str(query).strip() if query else ""
    if not clean_query:
        raise ValueError("Search query cannot be empty.")

    k = max(1, min(int(top_k), 10))
    chunks = retrieve_relevant_chunks(question=clean_query, top_k=k)

    return {
        "query": clean_query,
        "results": chunks,
        "count": len(chunks),
    }


def calculator_tool(a: float, b: float, operation: str) -> dict:
    """Execute a deterministic arithmetic operation on two numeric operands.

    Args:
        a: The first numerical operand.
        b: The second numerical operand.
        operation: The arithmetic operation ('add', 'subtract', 'multiply', 'divide').

    Returns:
        A dictionary containing the input operands, operation, and computed result.
    """
    try:
        num_a = float(a)
        num_b = float(b)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Operands 'a' and 'b' must be numbers. Received a={a}, b={b}: {exc}") from exc

    op_str = str(operation).strip().lower()
    res = base_calculator(a=num_a, b=num_b, operation=op_str)
    return {
        "a": num_a,
        "b": num_b,
        "operation": op_str,
        "result": res,
    }


def ask_user_clarification_tool(question: str) -> dict:
    """Ask the user a clarifying question when the original input is underspecified or ambiguous.

    Args:
        question: The concise clarifying question to ask the user.

    Returns:
        A dictionary signifying clarification is required along with the question text.
    """
    clean_q = str(question).strip() if question else ""
    if not clean_q:
        raise ValueError("Clarification question cannot be empty.")

    return {
        "clarification_required": True,
        "question": clean_q,
    }


class RegisteredTool(BaseModel):
    """Metadata and execution reference for a registered tool."""
    name: str
    description: str
    parameters: dict
    func: Any = Field(exclude=True)

    model_config = {"arbitrary_types_allowed": True}


class ToolRegistry:
    """Provider-agnostic tool registry managing tool definitions, validation, and safe dispatching."""

    def __init__(self):
        self._tools: Dict[str, RegisteredTool] = {}

    def register(
        self,
        name: str,
        func: Callable[..., Any],
        description: str,
        parameters: dict,
    ) -> None:
        """Register a new tool into the registry."""
        self._tools[name] = RegisteredTool(
            name=name,
            description=description,
            parameters=parameters,
            func=func,
        )
        logger.debug(f"Registered tool: '{name}'")

    def get_tool(self, name: str) -> Optional[RegisteredTool]:
        """Retrieve a registered tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[dict]:
        """Return a list of tool metadata dictionaries for prompt construction and schema declaration."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in self._tools.values()
        ]

    def execute(self, tool_name: str, arguments: dict) -> ToolResult:
        """Safely dispatch and execute a registered tool with argument validation and timing.

        Returns a standardized ToolResult contract, catching all exceptions as structured errors.
        """
        start_time = time.perf_counter()
        tool_obj = self.get_tool(tool_name)

        if not tool_obj:
            elapsed = time.perf_counter() - start_time
            return ToolResult(
                success=False,
                tool=tool_name,
                error=ToolError(
                    type="UnknownToolError",
                    message=f"Tool '{tool_name}' is not registered. Available tools: {list(self._tools.keys())}",
                ),
                execution_time_seconds=round(elapsed, 4),
            )

        try:
            kwargs = arguments or {}
            # Verify signature compatibility
            sig = inspect.signature(tool_obj.func)
            bound_args = sig.bind(**kwargs)
            bound_args.apply_defaults()

            result = tool_obj.func(*bound_args.args, **bound_args.kwargs)
            elapsed = time.perf_counter() - start_time
            return ToolResult(
                success=True,
                tool=tool_name,
                result=result,
                error=None,
                execution_time_seconds=round(elapsed, 4),
            )
        except TypeError as exc:
            elapsed = time.perf_counter() - start_time
            logger.warning(f"Invalid arguments for tool '{tool_name}': {exc}")
            return ToolResult(
                success=False,
                tool=tool_name,
                error=ToolError(
                    type="InvalidArgumentsError",
                    message=f"Invalid arguments provided to '{tool_name}': {str(exc)}",
                ),
                execution_time_seconds=round(elapsed, 4),
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            err_type = type(exc).__name__
            logger.warning(f"Tool execution failed for '{tool_name}' ({err_type}): {exc}")
            return ToolResult(
                success=False,
                tool=tool_name,
                error=ToolError(
                    type=err_type,
                    message=str(exc),
                ),
                execution_time_seconds=round(elapsed, 4),
            )


# Initialize and populate default singleton tool registry
default_tool_registry = ToolRegistry()

# 1. RAG Search Tool
default_tool_registry.register(
    name="rag_search",
    func=rag_search_tool,
    description=(
        "Dynamically search the knowledge base for relevant document chunks. "
        "Use this tool whenever you need factual information, document definitions, or algorithm details. "
        "You can formulate distinct search queries across multiple iterations if initial results are incomplete."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query, topic keyword, or question to search the knowledge base for.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of document chunks to retrieve (default: 3, max: 10).",
                "default": 3,
            },
        },
        "required": ["query"],
    },
)

# 2. Calculator Tool
default_tool_registry.register(
    name="calculator",
    func=calculator_tool,
    description=(
        "Perform exact arithmetic operations on two numbers. "
        "Supported operations: 'add', 'subtract', 'multiply', 'divide'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "First operand (number)."},
            "b": {"type": "number", "description": "Second operand (number)."},
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"],
                "description": "The mathematical operation to compute.",
            },
        },
        "required": ["a", "b", "operation"],
    },
)

# 3. Ask User Clarification Tool
default_tool_registry.register(
    name="ask_user_clarification",
    func=ask_user_clarification_tool,
    description=(
        "Request clarification from the user when the request is underspecified, "
        "contains ambiguous pronouns (e.g. 'its complexity' without context), or lacks essential parameters."
    ),
    parameters={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The concise clarification question to ask the user.",
            },
        },
        "required": ["question"],
    },
)
