"""booktabs LaTeX tables in the style of the paper (decimal comma)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def escape(s) -> str:
    return (str(s).replace("\\", "\\textbackslash{}").replace("_", "\\_")
            .replace("%", "\\%").replace("&", "\\&").replace("#", "\\#"))


def cell(v, decimals: int) -> str:
    if isinstance(v, (float, np.floating)):
        return "--" if np.isnan(v) else f"{v:.{decimals}f}".replace(".", ",")
    if isinstance(v, (int, np.integer)):
        return f"{v:,}".replace(",", ".")
    s = str(v)
    if re.fullmatch(r"[\d.\-\s\[\];±]+", s):  # numeric text: "0.712 [0.70; 0.72]"
        s = s.replace(".", ",")
    return escape(s).replace("±", "$\\pm$")


def table(df: pd.DataFrame, caption: str, label: str, decimals: int = 3, index_name: str = "Modelo") -> str:
    """Headers are raw LaTeX. Wide tables (> 5 columns) are resized to the text width."""
    head = " & ".join([f"\\textbf{{{index_name}}}"] + [f"\\textbf{{{c}}}" for c in df.columns])
    body = "\n".join(" & ".join([escape(i)] + [cell(v, decimals) for v in row]) + " \\\\"
                     for i, row in zip(df.index, df.itertuples(index=False)))
    tab = (f"\\begin{{tabular}}{{l{'r' * df.shape[1]}}}\n\\toprule\n{head} \\\\\n\\midrule\n{body}\n"
           f"\\bottomrule\n\\end{{tabular}}")
    if df.shape[1] > 5:
        tab = f"\\resizebox{{\\linewidth}}{{!}}{{%\n{tab}}}"
    return (f"\\begin{{table}}[ht]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\small\n"
            f"{tab}\n\\end{{table}}\n")
