"""Calculator tool for mathematical operations."""


def calculator(a: float, b: float, operation: str) -> float:
    """Perform a basic arithmetic operation on two numbers.

    Args:
        a: The first operand.
        b: The second operand.
        operation: The arithmetic operation to perform ('add', 'subtract', 'multiply', 'divide').

    Returns:
        The result of the arithmetic operation as a float.

    Raises:
        ValueError: If division by zero is attempted or an unsupported operation is specified.
    """
    op = operation.lower().strip()
    if op == "add":
        return float(a + b)
    elif op == "subtract":
        return float(a - b)
    elif op == "multiply":
        return float(a * b)
    elif op == "divide":
        if b == 0:
            raise ValueError("Division by zero is not allowed.")
        return float(a / b)
    else:
        raise ValueError(
            f"Unsupported operation '{operation}'. Supported operations: add, subtract, multiply, divide."
        )
