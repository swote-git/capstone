#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib").setLevel(logging.ERROR)
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns


def set_korean_font() -> str:
    candidates = ["NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "AppleGothic", "Malgun Gothic"]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), "DejaVu Sans")
    plt.rcParams["font.family"] = chosen
    plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fund analytics figures by 5 axes (A~E)")
    p.add_argument("--fund-raw-csv", type=Path, default=Path("data/12.금융상품정보/공모펀드상품.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("reports/product12/fund_advanced_figures"))
    return p.parse_args()


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")


def to_bin(s: pd.Series) -> pd.Series:
    x = s.replace({"Y": 1, "N": 0, "y": 1, "n": 0, True: 1, False: 0})
    x = pd.to_numeric(x, errors="coerce").fillna(0)
    return (x > 0).astype(int)


def _slice_rects(weights: np.ndarray, x: float, y: float, w: float, h: float, vertical: bool) -> list[tuple[float, float, float, float]]:
    rects: list[tuple[float, float, float, float]] = []
    total = float(weights.sum())
    if total <= 0:
        return rects
    cur_x, cur_y = x, y
    for ww in weights:
        area = (float(ww) / total) * w * h
        if vertical:
            rw = 0.0 if h <= 0 else area / h
            rects.append((cur_x, cur_y, rw, h))
            cur_x += rw
        else:
            rh = 0.0 if w <= 0 else area / w
            rects.append((cur_x, cur_y, w, rh))
            cur_y += rh
    return rects


def fig_a1_treemap(df: pd.DataFrame, out_dir: Path) -> None:
    g = (
        df.groupby(["대유형", "중유형", "소유형"], dropna=False)
        .size()
        .reset_index(name="cnt")
        .sort_values("cnt", ascending=False)
    )
    majors = g.groupby("대유형", dropna=False)["cnt"].sum().sort_values(ascending=False)
    major_rects = _slice_rects(majors.to_numpy(), 0, 0, 1, 1, vertical=True)

    fig, ax = plt.subplots(figsize=(14, 8))
    cmap = plt.get_cmap("tab20")

    for i, (maj_name, maj_cnt) in enumerate(majors.items()):
        mx, my, mw, mh = major_rects[i]
        sub = g[g["대유형"] == maj_name].copy()
        mids = sub.groupby("중유형", dropna=False)["cnt"].sum().sort_values(ascending=False)
        mid_rects = _slice_rects(mids.to_numpy(), mx, my, mw, mh, vertical=False)
        major_color = cmap(i % 20)

        for j, (mid_name, mid_cnt) in enumerate(mids.items()):
            sx, sy, sw, sh = mid_rects[j]
            leaf = sub[sub["중유형"] == mid_name].sort_values("cnt", ascending=False)
            leaf_rects = _slice_rects(leaf["cnt"].to_numpy(), sx, sy, sw, sh, vertical=True)
            for k, (_, r) in enumerate(leaf.reset_index(drop=True).iterrows()):
                lx, ly, lw, lh = leaf_rects[k]
                alpha = 0.35 + 0.45 * (k / max(1, len(leaf_rects) - 1))
                ax.add_patch(Rectangle((lx, ly), lw, lh, facecolor=major_color, edgecolor="white", linewidth=0.5, alpha=alpha))
                if lw * lh > 0.012:
                    ax.text(lx + lw * 0.02, ly + lh * 0.5, f"{r['소유형']}\n{int(r['cnt'])}", fontsize=7, va="center")

        if mw * mh > 0.03:
            ax.text(mx + mw * 0.01, my + mh * 0.98, f"{maj_name} ({int(maj_cnt)})", fontsize=10, va="top", fontweight="bold")

    ax.set_title("A-1. 대/중/소유형 Treemap (펀드 Universe)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "A1_treemap_major_mid_small.png", dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    font_name = set_korean_font()

    df = pd.read_csv(args.fund_raw_csv, low_memory=False)

    # basic typed columns
    num_cols = [
        "설정액", "순자산", "패밀리설정액", "패밀리순자산", "투자위험등급",
        "펀드성과정보_1개월", "펀드성과정보_3개월", "펀드성과정보_6개월", "펀드성과정보_1년",
        "펀드표준편차_1년", "펀드수정샤프_1년", "MaximumDrawDown_1년",
        "NetCashFlow(펀드자금흐름)_1개월", "NetCashFlow(펀드자금흐름)_3개월", "NetCashFlow(펀드자금흐름)_6개월", "NetCashFlow(펀드자금흐름)_1년",
        "FamilyNetCashFlow(패밀리펀드자금흐름)_1개월", "FamilyNetCashFlow(패밀리펀드자금흐름)_3개월", "FamilyNetCashFlow(패밀리펀드자금흐름)_6개월", "FamilyNetCashFlow(패밀리펀드자금흐름)_1년",
        "운용보수", "수탁보수", "사무관리보수", "판매보수", "선취수수료", "후취수수료",
        "1년종합등급", "3년종합등급", "5년종합등급",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = to_num(df[c])

    # tag columns (14)
    tag_cols = [
        "가치주", "성장주", "중소형주", "글로벌", "자산배분", "4차산업", "ESG(사회책임투자형)",
        "배당주", "FoFs", "퇴직연금", "고난도금융상품", "절대수익추구", "레버리지", "퀀트",
    ]
    existing_tags = [c for c in tag_cols if c in df.columns]
    for c in existing_tags:
        df[c] = to_bin(df[c])

    # A-1
    fig_a1_treemap(df, args.out_dir)

    # A-2 tag frequency
    tag_freq = df[existing_tags].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=tag_freq.values, y=tag_freq.index, color="#4C78A8", ax=ax)
    ax.set_title("A-2. 투자특성 태그 빈도")
    ax.set_xlabel("fund count")
    ax.set_ylabel("tag")
    fig.tight_layout(); fig.savefig(args.out_dir / "A2_tag_frequency_bar.png", dpi=170); plt.close(fig)

    # A-3 tag co-occurrence
    co = df[existing_tags].T.dot(df[existing_tags])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(co, cmap="YlGnBu", ax=ax)
    ax.set_title("A-3. 태그 Co-occurrence 히트맵")
    fig.tight_layout(); fig.savefig(args.out_dir / "A3_tag_cooccurrence_heatmap.png", dpi=170); plt.close(fig)

    # A-4 manager market share
    m = (
        df.groupby("운용사명", dropna=False)
        .agg(fund_count=("펀드코드", "count"), setting_amt=("설정액", "sum"))
        .sort_values("setting_amt", ascending=False)
        .head(15)
        .reset_index()
    )
    m["setting_share"] = m["setting_amt"] / max(m["setting_amt"].sum(), 1)
    m["count_share"] = m["fund_count"] / max(m["fund_count"].sum(), 1)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    sns.barplot(data=m, x="운용사명", y="setting_share", color="#59A14F", ax=ax1)
    ax1.tick_params(axis="x", rotation=60)
    ax1.set_ylabel("설정액 비중")
    ax1.set_title("A-4. 운용사별 펀드 수 / 설정액 비중 (Top15)")
    ax2 = ax1.twinx()
    ax2.plot(range(len(m)), m["count_share"], color="#E15759", marker="o")
    ax2.set_ylabel("펀드 수 비중")
    fig.tight_layout(); fig.savefig(args.out_dir / "A4_manager_market_share.png", dpi=170); plt.close(fig)

    # B-1 risk-return scatter
    b = df[["대유형", "펀드표준편차_1년", "펀드성과정보_1년"]].dropna()
    top_major = b["대유형"].astype(str).value_counts().head(8).index
    b = b[b["대유형"].astype(str).isin(top_major)].copy()
    major_order = [str(x) for x in top_major]
    major_colors = sns.color_palette("tab10", n_colors=len(major_order))
    major_palette = {m: major_colors[i] for i, m in enumerate(major_order)}

    xcol = "펀드표준편차_1년"
    ycol = "펀드성과정보_1년"
    x_lo, x_hi = b[xcol].quantile(0.01), b[xcol].quantile(0.99)
    y_lo, y_hi = b[ycol].quantile(0.01), b[ycol].quantile(0.99)

    core_mask = b[xcol].between(x_lo, x_hi) & b[ycol].between(y_lo, y_hi)
    core = b[core_mask]
    outlier = b[~core_mask]

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=core,
        x=xcol,
        y=ycol,
        hue="대유형",
        hue_order=major_order,
        palette=major_palette,
        alpha=0.60,
        s=22,
        ax=ax,
    )
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_title("B-1. 리스크-수익 산점도 (중심 98% 확대)")
    ax.set_xlabel("펀드표준편차_1년")
    ax.set_ylabel("펀드성과정보_1년")
    ax.text(
        0.98,
        0.98,
        f"전체: {len(b):,}개\n중심: {len(core):,}개\n아웃라이어 제외: {len(outlier):,}개",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F8F9F9", "edgecolor": "#BFC9CA", "alpha": 0.95},
    )
    ax.legend(title="대유형", loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(args.out_dir / "B1_risk_return_scatter.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # B-2 sharpe distribution layered
    b2 = df[["대유형", "펀드수정샤프_1년"]].dropna().copy()
    p99_sharpe = b2["펀드수정샤프_1년"].quantile(0.99)
    b2 = b2[b2["펀드수정샤프_1년"] <= p99_sharpe]
    top_major2 = b2["대유형"].astype(str).value_counts().head(6).index
    fig, ax = plt.subplots(figsize=(10, 6))
    for major in top_major2:
        vals = b2.loc[b2["대유형"].astype(str) == str(major), "펀드수정샤프_1년"]
        sns.kdeplot(vals, label=str(major), ax=ax, linewidth=1.8, color=major_palette.get(str(major), None))
    ax.set_title("B-2. 수정샤프비율 분포 (유형별, p99 필터)")
    ax.text(
        0.02,
        0.98,
        f"p99={p99_sharpe:.3f}, n={len(b2):,}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(args.out_dir / "B2_sharpe_distribution_layered.png", dpi=170); plt.close(fig)

    # B-3 MDD vs sharpe
    b3 = df[["MaximumDrawDown_1년", "펀드수정샤프_1년", "대유형"]].dropna().copy()
    mdd_p1, mdd_p99 = b3["MaximumDrawDown_1년"].quantile([0.01, 0.99]).tolist()
    sh_p1, sh_p99 = b3["펀드수정샤프_1년"].quantile([0.01, 0.99]).tolist()
    b3 = b3[
        b3["MaximumDrawDown_1년"].between(mdd_p1, mdd_p99)
        & b3["펀드수정샤프_1년"].between(sh_p1, sh_p99)
    ]
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=b3,
        x="MaximumDrawDown_1년",
        y="펀드수정샤프_1년",
        hue="대유형",
        hue_order=major_order,
        palette=major_palette,
        alpha=0.5,
        ax=ax,
    )
    ax.set_title("B-3. MDD vs 수정샤프 (p1~p99 필터)")
    ax.text(
        0.02,
        0.98,
        f"MDD p1~p99: {mdd_p1:.3f}~{mdd_p99:.3f}\n샤프 p1~p99: {sh_p1:.3f}~{sh_p99:.3f}\nn={len(b3):,}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )
    ax.legend(title="대유형", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=8)
    fig.tight_layout(); fig.savefig(args.out_dir / "B3_mdd_vs_sharpe_scatter.png", dpi=170, bbox_inches="tight"); plt.close(fig)

    color_legend = pd.DataFrame(
        {"대유형": major_order, "color_hex": [to_hex(major_palette[m]) for m in major_order]}
    )
    color_legend.to_csv(args.out_dir / "B_color_legend.csv", index=False, encoding="utf-8-sig")

    # B-4 multi-period returns correlation
    rcols = ["펀드성과정보_1개월", "펀드성과정보_3개월", "펀드성과정보_6개월", "펀드성과정보_1년"]
    corr = df[rcols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
    ax.set_title("B-4. 다기간 수익률 상관행렬")
    fig.tight_layout(); fig.savefig(args.out_dir / "B4_return_horizon_correlation.png", dpi=170); plt.close(fig)

    # C-1 1/3/5Y grade consistency heatmap (1Y vs 5Y)
    # Use only 1~5 grades; exclude 0 as likely unrated / insufficient 5Y history.
    c1 = df[["1년종합등급", "3년종합등급", "5년종합등급"]].dropna().copy()
    c1["1년종합등급"] = c1["1년종합등급"].round().astype(int)
    c1["5년종합등급"] = c1["5년종합등급"].round().astype(int)
    c1 = c1[c1["1년종합등급"].between(1, 5) & c1["5년종합등급"].between(1, 5)]
    gmat = pd.crosstab(c1["1년종합등급"], c1["5년종합등급"]).reindex(index=[1, 2, 3, 4, 5], columns=[1, 2, 3, 4, 5], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(gmat, annot=True, fmt="d", cmap="PuBu", ax=ax)
    ax.set_title("C-1. 1년 vs 5년 등급 일관성 (1~5 등급만)")
    ax.text(
        0.02,
        0.98,
        "0등급 제외: 5년 등급 산출 이력 미충분 가능성",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f8f9fa", "edgecolor": "#bdc3c7", "alpha": 0.95},
    )
    fig.tight_layout(); fig.savefig(args.out_dir / "C1_grade_consistency_heatmap.png", dpi=170); plt.close(fig)

    # C-2 risk-grade distribution
    c2 = df["투자위험등급"].dropna().round().astype(int)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x=c2, color="#F28E2B", ax=ax)
    ax.set_title("C-2. 투자위험등급 분포")
    ax.set_xlabel("risk grade")
    ax.set_ylabel("count")
    fig.tight_layout(); fig.savefig(args.out_dir / "C2_investment_risk_grade_distribution.png", dpi=170); plt.close(fig)

    # C-3 net asset distribution (log)
    c3 = df["순자산"].dropna()
    c3 = c3[c3 > 0]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(np.log10(c3), bins=50, color="#B07AA1", ax=ax)
    ax.set_title("C-3. 순자산 분포 (log10)")
    ax.set_xlabel("log10(순자산)")
    fig.tight_layout(); fig.savefig(args.out_dir / "C3_net_asset_log_distribution.png", dpi=170); plt.close(fig)

    # C-4 setting vs net-asset ratio
    c4 = df[["설정액", "순자산"]].dropna()
    c4 = c4[(c4["설정액"] > 0) & (c4["순자산"] > 0)].copy()
    c4["asset_to_setting_ratio"] = c4["순자산"] / c4["설정액"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=c4.sample(n=min(len(c4), 5000), random_state=42), x="설정액", y="asset_to_setting_ratio", alpha=0.4, ax=ax)
    ax.set_xscale("log")
    ax.set_title("C-4. 설정액 vs 순자산/설정액 비율")
    ax.set_xlabel("설정액 (log)")
    ax.set_ylabel("순자산/설정액")
    fig.tight_layout(); fig.savefig(args.out_dir / "C4_setting_vs_asset_ratio.png", dpi=170); plt.close(fig)

    # D-1 NCF direction ratios
    dcols = [
        "NetCashFlow(펀드자금흐름)_1개월", "NetCashFlow(펀드자금흐름)_3개월", "NetCashFlow(펀드자금흐름)_6개월", "NetCashFlow(펀드자금흐름)_1년"
    ]
    ratio_rows = []
    for c in dcols:
        s = df[c].dropna()
        n = max(len(s), 1)
        ratio_rows.append({"horizon": c.split("_")[-1], "유입(+)": float((s > 0).sum()) / n, "유출(-)": float((s < 0).sum()) / n, "중립(0)": float((s == 0).sum()) / n})
    dr = pd.DataFrame(ratio_rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(dr))
    for col, color in [("유입(+)", "#59A14F"), ("유출(-)", "#E15759"), ("중립(0)", "#BAB0AC")]:
        ax.bar(dr["horizon"], dr[col], bottom=bottom, label=col, color=color)
        bottom += dr[col].to_numpy()
    ax.set_title("D-1. NCF 방향성 비율")
    ax.set_ylabel("ratio")
    ax.legend()
    fig.tight_layout(); fig.savefig(args.out_dir / "D1_ncf_direction_ratio.png", dpi=170); plt.close(fig)

    # D-2 NCF vs return
    d2 = df[["NetCashFlow(펀드자금흐름)_1년", "펀드성과정보_1년", "대유형"]].dropna()
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(data=d2.sample(n=min(len(d2), 6000), random_state=42), x="NetCashFlow(펀드자금흐름)_1년", y="펀드성과정보_1년", hue="대유형", alpha=0.55, ax=ax)
    ax.set_xscale("symlog")
    ax.set_title("D-2. NCF(1Y) vs 수익률(1Y)")
    fig.tight_layout(); fig.savefig(args.out_dir / "D2_ncf_vs_return_scatter.png", dpi=170); plt.close(fig)

    # D-3 family ncf vs single ncf
    d3 = df[["NetCashFlow(펀드자금흐름)_1년", "FamilyNetCashFlow(패밀리펀드자금흐름)_1년"]].dropna()
    d3s = d3.sample(n=min(len(d3), 6000), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.scatterplot(data=d3s, x="NetCashFlow(펀드자금흐름)_1년", y="FamilyNetCashFlow(패밀리펀드자금흐름)_1년", alpha=0.45, ax=ax)
    lim = np.nanmax(np.abs(d3s.to_numpy()))
    lim = float(lim) if np.isfinite(lim) and lim > 0 else 1.0
    ax.plot([-lim, lim], [-lim, lim], color="red", linestyle="--", linewidth=1)
    ax.set_xscale("symlog"); ax.set_yscale("symlog")
    ax.set_title("D-3. Family NCF vs 단일 NCF (1Y)")
    fig.tight_layout(); fig.savefig(args.out_dir / "D3_family_vs_single_ncf.png", dpi=170); plt.close(fig)

    # E-1 total fee distribution
    e = df[["운용보수", "수탁보수", "판매보수", "사무관리보수", "대유형"]].copy()
    e["총보수"] = e[["운용보수", "수탁보수", "판매보수", "사무관리보수"]].sum(axis=1, skipna=True)
    e = e.dropna(subset=["총보수", "대유형"])
    topm = e["대유형"].astype(str).value_counts().head(6).index
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=e[e["대유형"].astype(str).isin(topm)], x="대유형", y="총보수", ax=ax)
    ax.tick_params(axis="x", rotation=35)
    ax.set_title("E-1. 총보수 분포 (유형별)")
    fig.tight_layout(); fig.savefig(args.out_dir / "E1_total_fee_distribution.png", dpi=170); plt.close(fig)

    # E-2 front/back fee combination heatmap
    e2 = df[["선취수수료", "후취수수료"]].copy().dropna()
    e2["선취_bin"] = pd.cut(e2["선취수수료"], bins=[-1e-9, 0, 0.5, 1.0, np.inf], labels=["0", "0~0.5", "0.5~1.0", ">1.0"])
    e2["후취_bin"] = pd.cut(e2["후취수수료"], bins=[-1e-9, 0, 0.5, 1.0, np.inf], labels=["0", "0~0.5", "0.5~1.0", ">1.0"])
    hm = pd.crosstab(e2["선취_bin"], e2["후취_bin"])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(hm, annot=True, fmt="d", cmap="Oranges", ax=ax)
    ax.set_title("E-2. 선취/후취 수수료 조합")
    fig.tight_layout(); fig.savefig(args.out_dir / "E2_front_back_fee_heatmap.png", dpi=170); plt.close(fig)

    # E-3 total fee vs return/sharpe
    e3 = df[["운용보수", "수탁보수", "판매보수", "사무관리보수", "펀드성과정보_1년", "펀드수정샤프_1년"]].copy()
    e3["총보수"] = e3[["운용보수", "수탁보수", "판매보수", "사무관리보수"]].sum(axis=1, skipna=True)
    e3 = e3.dropna(subset=["총보수", "펀드성과정보_1년", "펀드수정샤프_1년"])
    e3s = e3.sample(n=min(len(e3), 7000), random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.scatterplot(data=e3s, x="총보수", y="펀드성과정보_1년", alpha=0.35, ax=axes[0])
    axes[0].set_title("E-3a. 총보수 vs 1Y 수익률")
    sns.scatterplot(data=e3s, x="총보수", y="펀드수정샤프_1년", alpha=0.35, ax=axes[1])
    axes[1].set_title("E-3b. 총보수 vs 1Y 수정샤프")
    fig.tight_layout(); fig.savefig(args.out_dir / "E3_total_fee_vs_return_sharpe.png", dpi=170); plt.close(fig)

    # README
    lines = [
        "# 펀드 5개 축 분석 Figures",
        "",
        f"- 적용 폰트: {font_name}",
        "- 축 A~E 기준 생성",
        "",
        "## A. 상품 분류 구조",
        "- A1_treemap_major_mid_small.png",
        "- A2_tag_frequency_bar.png",
        "- A3_tag_cooccurrence_heatmap.png",
        "- A4_manager_market_share.png",
        "",
        "## B. 리스크-수익 프로파일",
        "- B1_risk_return_scatter.png",
        "- B2_sharpe_distribution_layered.png",
        "- B3_mdd_vs_sharpe_scatter.png",
        "- B4_return_horizon_correlation.png",
        "- B_color_legend.csv (대유형-색상 매핑)",
        "",
        "## C. 등급 및 규모 검증",
        "- C1_grade_consistency_heatmap.png",
        "- C2_investment_risk_grade_distribution.png",
        "- C3_net_asset_log_distribution.png",
        "- C4_setting_vs_asset_ratio.png",
        "",
        "## D. 자금흐름 (NCF)",
        "- D1_ncf_direction_ratio.png",
        "- D2_ncf_vs_return_scatter.png",
        "- D3_family_vs_single_ncf.png",
        "",
        "## E. 비용 구조",
        "- E1_total_fee_distribution.png",
        "- E2_front_back_fee_heatmap.png",
        "- E3_total_fee_vs_return_sharpe.png",
    ]
    (args.out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"saved figures dir: {args.out_dir}")


if __name__ == "__main__":
    main()
