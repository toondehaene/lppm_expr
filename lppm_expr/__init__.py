# lppm_expr/__init__.py
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function
from polars._typing import IntoExpr

PLUGIN_PATH = Path(__file__).parent


def stateful_acc(expr: IntoExpr) -> pl.Expr:
    """Stateful accumulator expression."""
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="stateful_acc",
        args=expr,
        is_elementwise=False,
        changes_length=False,
        returns_scalar=False,
        cast_to_supertype=False,
        input_wildcard_expansion=False,
        pass_name_to_apply=False,
        use_abs_path=False,
    )

def cum_sum_weighted(expr: IntoExpr) -> pl.Expr:
    return register_plugin_function(
        args=expr,
        plugin_path=PLUGIN_PATH,
        function_name="cum_sum_weighted",
        is_elementwise=False,
        changes_length=False,
        returns_scalar=False,
        cast_to_supertype=False,
        input_wildcard_expansion=False,
        pass_name_to_apply=False,
        use_abs_path=False,
    )
