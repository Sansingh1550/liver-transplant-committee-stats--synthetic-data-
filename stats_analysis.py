import itertools
import numpy as np
import pandas as pd
import sklearn.metrics
import statsmodels.stats.proportion as prop
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests


def ppv_npv_with_ci(label, vote, alpha=0.05):
    label = np.asarray(label).astype(int)
    vote = np.asarray(vote).astype(int)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(
        label, vote, labels=[0, 1]
    ).ravel().tolist()
    n_pred_pos = tp + fp
    n_pred_neg = tn + fn

    def wilson(count, total):
        if total == 0:
            return (np.nan, np.nan, np.nan)
        p_hat = count / total
        lo, hi = prop.proportion_confint(count, total, alpha=alpha, method="wilson")
        return (p_hat, lo, hi)

    return {
        "n": tp + tn + fp + fn, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "ppv": wilson(tp, n_pred_pos),
        "npv": wilson(tn, n_pred_neg),
    }


def build_ppv_npv_table(df, game_cols):
    rows = []
    for game_name in game_cols:
        m = ppv_npv_with_ci(df["label"], df[f"vote_{game_name}"])
        row = {"game": game_name, "n": m["n"],
               "tp": m["tp"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"]}
        for metric in ["ppv", "npv"]:
            pt, lo, hi = m[metric]
            row[metric] = pt
            row[f"{metric}_ci"] = f"[{lo:.3f}, {hi:.3f}]" if not np.isnan(lo) else "NA"
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_mcnemar(df, game_cols, correction_method="holm"):
    results = []
    for game_a, game_b in itertools.combinations(game_cols, 2):
        correct_a = (df[f"vote_{game_a}"] == df["label"]).astype(int)
        correct_b = (df[f"vote_{game_b}"] == df["label"]).astype(int)
        table = pd.crosstab(correct_a, correct_b).reindex(
            index=[0, 1], columns=[0, 1], fill_value=0
        ).values
        both_wrong, a_wrong_b_right = table[0, 0], table[0, 1]
        a_right_b_wrong, both_right = table[1, 0], table[1, 1]
        res = mcnemar(table, exact=True)
        results.append({
            "game_a": game_a, "game_b": game_b,
            "n_both_right": both_right, "n_both_wrong": both_wrong,
            "n_a_right_only": a_right_b_wrong, "n_b_right_only": a_wrong_b_right,
            "p_value_raw": res.pvalue,
        })
    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        reject, p_adj, _, _ = multipletests(results_df["p_value_raw"], method=correction_method)
        results_df["p_value_holm"] = p_adj
        results_df["significant_after_correction"] = reject
    return results_df


def _metric_value(label, vote, metric):
    label = np.asarray(label).astype(int)
    vote = np.asarray(vote).astype(int)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(
        label, vote, labels=[0, 1]
    ).ravel().tolist()
    if metric == "accuracy":
        denom = tp + tn + fp + fn
        return (tp + tn) / denom if denom else np.nan
    if metric == "sensitivity":
        denom = tp + fn
        return tp / denom if denom else np.nan
    if metric == "specificity":
        denom = tn + fp
        return tn / denom if denom else np.nan
    if metric == "ppv":
        denom = tp + fp
        return tp / denom if denom else np.nan
    if metric == "npv":
        denom = tn + fn
        return tn / denom if denom else np.nan
    if metric == "f1":
        precision = _metric_value(label, vote, "ppv")
        recall = _metric_value(label, vote, "sensitivity")
        if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
            return np.nan
        return 2 * precision * recall / (precision + recall)
    raise ValueError(f"Unknown metric: {metric}")


def paired_bootstrap_diff(df, game_a, game_b, metric="accuracy", n_boot=5000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = len(df)
    label = df["label"].values
    vote_a = df[f"vote_{game_a}"].values
    vote_b = df[f"vote_{game_b}"].values
    observed_diff = (_metric_value(label, vote_a, metric) - _metric_value(label, vote_b, metric))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = (_metric_value(label[idx], vote_a[idx], metric) - _metric_value(label[idx], vote_b[idx], metric))
    lo, hi = np.nanpercentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "game_a": game_a, "game_b": game_b, "metric": metric,
        "observed_diff": observed_diff, "ci_low": lo, "ci_high": hi,
        "ci_excludes_zero": (lo > 0) or (hi < 0),
    }


def all_pairwise_bootstrap(df, game_cols, metrics=("accuracy", "sensitivity", "specificity", "f1"), n_boot=5000):
    rows = []
    for game_a, game_b in itertools.combinations(game_cols, 2):
        for metric in metrics:
            rows.append(paired_bootstrap_diff(df, game_a, game_b, metric, n_boot))
    return pd.DataFrame(rows)


def error_type(label, vote):
    if label == 1 and vote == 1:
        return "TP"
    if label == 0 and vote == 0:
        return "TN"
    if label == 0 and vote == 1:
        return "FP"
    return "FN"


def subgroup_vs_rest_test(df, demo_col, level, game_name, error_col="FP"):
    in_group = df[demo_col] == level
    err_flags = (df[f"err_{game_name}"] == error_col).astype(int)
    eligible_label = 0 if error_col == "FP" else 1
    eligible = df["label"] == eligible_label
    g_count = int((err_flags[in_group & eligible]).sum())
    g_n = int((in_group & eligible).sum())
    rest_count = int((err_flags[~in_group & eligible]).sum())
    rest_n = int((~in_group & eligible).sum())
    if g_n == 0 or rest_n == 0:
        return None
    stat, pval = proportions_ztest(count=[g_count, rest_count], nobs=[g_n, rest_n])
    return {
        "demo_col": demo_col, "level": level, "game": game_name, "error_type": error_col,
        "group_rate": g_count / g_n, "rest_rate": rest_count / rest_n, "p_value": pval,
    }


def run_subgroup_fairness_screen(df, game_cols, demo_cols):
    for game_name in game_cols:
        df[f"err_{game_name}"] = [error_type(l, v) for l, v in zip(df["label"], df[f"vote_{game_name}"])]
    results = []
    for demo_col in demo_cols:
        for level in df[demo_col].dropna().unique():
            for game_name in game_cols:
                for error_col in ["FP", "FN"]:
                    r = subgroup_vs_rest_test(df, demo_col, level, game_name, error_col)
                    if r is not None:
                        results.append(r)
    return pd.DataFrame(results).sort_values("p_value")


def add_fdr_correction(results_df, p_col="p_value", method="fdr_bh"):
    out = results_df.copy()
    valid = out[p_col].notna()
    reject, p_adj, _, _ = multipletests(out.loc[valid, p_col], method=method)
    out.loc[valid, "p_value_fdr"] = p_adj
    out.loc[valid, "significant_after_fdr"] = reject
    out["significant_after_fdr"] = out["significant_after_fdr"].fillna(False)
    return out.sort_values("p_value_fdr")