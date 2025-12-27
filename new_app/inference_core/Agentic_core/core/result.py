"""
Robust Result Type for Error Handling.

Adheres to:
- Failure Domain Analysis: Return Result/Either types; forbid exceptions for control flow.
- Zero-Cost: Minimal wrapper overhead.
"""
from typing import TypeVar, Generic, Union, Optional, Callable, Any
from dataclasses import dataclass

T = TypeVar("T")
E = TypeVar("E")

@dataclass(frozen=True)
class Ok(Generic[T]):
    value: T
    is_ok: bool = True
    is_err: bool = False

@dataclass(frozen=True)
class Err(Generic[E]):
    error: E
    is_ok: bool = False
    is_err: bool = True

Result = Union[Ok[T], Err[E]]

class ResultExt:
    """Extension methods for Result processing."""
    
    @staticmethod
    def map(result: Result[T, E], func: Callable[[T], Any]) -> Result[Any, E]:
        if result.is_ok:
            try:
                return Ok(func(result.value))
            except Exception as e:
                # Catch untrapped exceptions in mapping functions to maintain safety
                return Err(e)
        return result

    @staticmethod
    def unwrap(result: Result[T, E]) -> T:
        if result.is_ok:
            return result.value
        raise ValueError(f"Called unwrap on Err: {result.error}")

    @staticmethod
    def unwrap_or(result: Result[T, E], default: T) -> T:
        if result.is_ok:
            return result.value
        return default
