"""Aggregate per-tree volume predictions to plot/stand level (the Tier-B-comparable view).

Caveat: plot Vol_ha in the dataset is model-derived (Campos H + Schumacher V), so plot-level comparison
is semi-circular. The only hard ground truth is the cubed trees.
"""
import pandas as pd

def aggregate_to_plots(tree_df: pd.DataFrame, group_col: str = "plot",
                       vol_col: str = "pred_V") -> pd.DataFrame:
    """Sum per-tree predicted volume within each group. Returns group_col, sum_V, n_trees."""
    g = tree_df.groupby(group_col)[vol_col]
    out = g.agg(sum_V="sum", n_trees="count").reset_index()
    return out
