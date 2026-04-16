from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import pandas as pd


def split_users(snapshots: pd.DataFrame, user_col: str, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    users = snapshots[user_col].drop_duplicates().sample(frac=1.0, random_state=42)
    cutoff = max(1, int(len(users) * train_ratio))
    train_users = set(users.iloc[:cutoff])
    train_df = snapshots[snapshots[user_col].isin(train_users)].copy()
    eval_df = snapshots[~snapshots[user_col].isin(train_users)].copy()
    if eval_df.empty:
        eval_df = train_df.copy()
    return train_df, eval_df


def build_recommender_eval_report(
    rec: Any,
    snapshots: pd.DataFrame,
    user_key: str,
    ks: Sequence[int],
    max_eval_users: int,
) -> Dict[str, object]:
    train_df, eval_df = split_users(snapshots, user_col=user_key)
    report = {
        "snapshot_quality": rec.snapshot_quality_report(snapshots),
        "split": {
            "train_rows": int(len(train_df)),
            "eval_rows": int(len(eval_df)),
            "train_users": int(train_df[user_key].nunique()),
            "eval_users": int(eval_df[user_key].nunique()),
        },
        "train_df": train_df,
        "eval_df": eval_df,
        "evaluation": rec.evaluate(eval_df, ks=ks, max_users=max_eval_users),
    }
    return report

