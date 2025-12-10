# lppm_expr/__init__.py
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function
from polars._typing import IntoExpr, IntoExprColumn

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


def vertical_scan(expr: IntoExpr) -> pl.Expr:
    return register_plugin_function(
        args=expr,
        plugin_path=PLUGIN_PATH,
        function_name="vertical_scan",
        is_elementwise=False,
        changes_length=False,
        returns_scalar=False,
        cast_to_supertype=False,
        input_wildcard_expansion=False,
        pass_name_to_apply=False,
        use_abs_path=False,
    )


def lazy_fill_random(expr: IntoExpr) -> pl.Expr:
    return register_plugin_function(
        args=expr,
        plugin_path=PLUGIN_PATH,
        function_name="lazy_fill_random",
        is_elementwise=True,
        changes_length=False,
        returns_scalar=False,
        cast_to_supertype=False,
        input_wildcard_expansion=False,
        pass_name_to_apply=False,
        use_abs_path=False,
    )


def is_social_link(
    row_idx: IntoExpr,
    user_id: IntoExpr,
    lon_rad: IntoExpr,
    lat_rad: IntoExpr,
    event_start: IntoExpr,
    event_end: IntoExpr,
    offset: IntoExpr,
    *,
    threshold: float,
) -> pl.Expr:
    return register_plugin_function(
        args=[
            row_idx,
            user_id,
            lon_rad,
            lat_rad,
            event_start,
            event_end,
            offset,
        ],
        kwargs={"threshold": threshold},
        plugin_path=PLUGIN_PATH,
        function_name="is_social_link",
        is_elementwise=False,
        changes_length=True,
        returns_scalar=False,
        cast_to_supertype=False,
        input_wildcard_expansion=False,
        pass_name_to_apply=False,
        use_abs_path=False,
    )
