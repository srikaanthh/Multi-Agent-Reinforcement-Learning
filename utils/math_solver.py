from __future__ import annotations

from functools import reduce
from typing import Optional, Tuple
import operator
import re


_ARITHMETIC_EXPR_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)")
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

_MULTIPLY_KEYWORDS = ("each", "every", "times", "rows of", "groups of", "per ")
_DIVIDE_KEYWORDS = ("split", "divided", "equally", "per person", "average")
_SUBTRACT_KEYWORDS = ("left", "remain", "after", "gave", "lost", "spent", "difference")
_ADD_KEYWORDS = ("total", "altogether", "in all", "combined", "sum")


def extract_numeric_token(text: str) -> Optional[str]:
    match = _NUMBER_RE.search(text.replace("$", ""))
    if not match:
        return None
    return match.group(0).replace(",", "")


def format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def solve_math_question(question: str) -> Tuple[Optional[float], str]:
    question_lower = question.lower()
    expression_match = _ARITHMETIC_EXPR_RE.search(question_lower)
    if expression_match:
        left = float(expression_match.group(1))
        operator_symbol = expression_match.group(2)
        right = float(expression_match.group(3))
        result = _apply_operator(left, operator_symbol, right)
        if result is not None:
            return result, f"Detected arithmetic expression {format_number(left)} {operator_symbol} {format_number(right)}."

    numbers = [float(number.replace(",", "")) for number in _NUMBER_RE.findall(question_lower)]
    if len(numbers) < 2:
        return None, "Not enough numeric evidence to infer a result."

    if any(keyword in question_lower for keyword in _DIVIDE_KEYWORDS) and numbers[1] != 0:
        result = numbers[0] / numbers[1]
        return result, f"Inferred division from the prompt: {format_number(numbers[0])} / {format_number(numbers[1])}."

    if any(keyword in question_lower for keyword in _MULTIPLY_KEYWORDS):
        result = reduce(operator.mul, numbers, 1.0)
        return result, f"Inferred multiplication over extracted quantities: {' * '.join(format_number(n) for n in numbers)}."

    if any(keyword in question_lower for keyword in _SUBTRACT_KEYWORDS):
        result = numbers[0] - sum(numbers[1:])
        return result, f"Inferred subtraction from the lead quantity: {format_number(numbers[0])} - {format_number(sum(numbers[1:]))}."

    if any(keyword in question_lower for keyword in _ADD_KEYWORDS):
        result = sum(numbers)
        return result, f"Inferred addition over extracted quantities: {' + '.join(format_number(n) for n in numbers)}."

    result = sum(numbers[:2])
    return result, f"Defaulted to adding the first two extracted quantities: {format_number(numbers[0])} + {format_number(numbers[1])}."


def _apply_operator(left: float, operator_symbol: str, right: float) -> Optional[float]:
    if operator_symbol == "+":
        return left + right
    if operator_symbol == "-":
        return left - right
    if operator_symbol == "*":
        return left * right
    if operator_symbol == "/" and right != 0:
        return left / right
    return None
