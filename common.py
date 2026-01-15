from typing import Any
import typer


def opt(default: Any, help: str):
    return typer.Option(default, help=help)
