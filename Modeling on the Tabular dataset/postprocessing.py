"""
postprocessing.py — survival model visualisation and diagnostic utilities
for the ADNI tabular survival analysis pipeline.

All plot functions save figures to ``FIG_DIR`` (configured in ``config.py``)
and display them inline when running in a Jupyter notebook.  Filenames are
constructed from ``model_name`` and ``cohort`` arguments so that outputs from
different models and cohorts do not overwrite each other.

Visualisation categories:
  - Kaplan-Meier curves (by risk quartile, by binary subgroup)
  - Individual and group survival curve panels
  - Feature importance (native, signed Cox coefficients, permutation)
  - Overfitting diagnostics (learning curves, OOF vs. train vs. test summary)
  - Bootstrap confidence interval plots (violin, histogram, comparison bar)
  - Fixed-horizon ROC curves
  - Time-dependent Brier score and integrated Brier score
  - Calibration plots (predicted vs. KM-observed event probability)
  - Optuna HPO optimisation history
  - Subject survival curve grid (random sample)
"""
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import pandas as pd
from modeling import binary_horizon_dataset
from config import RANDOM_SEED, N_FOLDS, HORIZONS, FIG_DIR, CHECKPOINT_DIR, OUT_DIR, MRI_HARMONIZE_COLS, BASE_DIR
import itertools
from typing import List, Optional, Dict
from scipy import stats as scipy_stats

# ── shared style ─────────────────────────────────────────────────────────────
PALETTE   = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
GRAY_GRID = '#E8E8E8'
BG        = '#FAFAF9'

def _style_ax(ax):
    """Apply a consistent light background + subtle grid style to a matplotlib Axes.

    Args:
        ax (matplotlib.axes.Axes): The axes object to style in-place.

    Returns:
        None.  Modifies ``ax`` directly.
    """
    ax.set_facecolor(BG)
    ax.grid(True, color=GRAY_GRID, linewidth=0.7, zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor('#D0D0D0')


# ═══════════════════════════════════════════════════════════════════════════════
# Kaplan-Meier curves
# ═══════════════════════════════════════════════════════════════════════════════

def km_risk_quartile(risk_scores, y_event, y_duration, model_name, cohort,
                     log_rank=True):
    """
    Kaplan-Meier curves stratified by predicted risk quartile, with optional
    log-rank test p-value annotation.

    Quartiles with fewer than 5 subjects are silently skipped.

    Args:
        risk_scores (np.ndarray): Continuous risk scores (higher = higher risk).
        y_event (np.ndarray): Binary event indicators.
        y_duration (np.ndarray): Time to event or censoring in years.
        model_name (str): Used in the plot title and output filename.
        cohort (str): Cohort label (e.g. 'MCI->Dementia').
        log_rank (bool): Whether to annotate with the overall log-rank p-value.

    Saves figure to FIG_DIR as 'km_quartile_<model>_<cohort>.png'.
    """
    quartile = pd.qcut(risk_scores, 4, labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'],
                       duplicates='drop')
    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)
    kmf = KaplanMeierFitter()

    masks = {}
    for q, col in zip(['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'], colors):
        mask = (quartile == q).values
        if mask.sum() < 5:
            continue
        masks[q] = mask
        kmf.fit(y_duration[mask], event_observed=y_event[mask], label=f'{q} (n={mask.sum()})')
        kmf.plot_survival_function(ax=ax, color=col, ci_show=True, ci_alpha=0.15, linewidth=2)

    # Log-rank test annotation
    if log_rank and len(masks) >= 2:
        groups = [y_duration[m] for m in masks.values()]
        events = [y_event[m]   for m in masks.values()]
        try:
            result = multivariate_logrank_test(
                np.concatenate(groups),
                np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)]),
                event_observed=np.concatenate(events)
            )
            p = result.p_value
            p_str = f'p = {p:.4f}' if p >= 0.0001 else 'p < 0.0001'
            ax.text(0.97, 0.97, f'Log-rank {p_str}',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=10, style='italic',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc', alpha=0.8))
        except Exception:
            pass

    ax.set(xlabel='Years from Baseline', ylabel='P(No Event)',
           title=f'KM by Risk Quartile: {model_name} [{cohort}]', ylim=(0, 1.02))
    ax.axhline(0.5, color='gray', ls=':', alpha=0.6, linewidth=1)
    ax.legend(fontsize=9, framealpha=0.9)
    plt.tight_layout()

    cohort_clean = cohort.replace('>', '')
    fname = FIG_DIR / f'km_quartile_{model_name.replace(" ", "_")}_{cohort_clean}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {fname}')


def km_binary_comparison(y_event, y_duration, group_mask, group_labels,
                          title, fname_stem, cohort=''):
    """
    Plot Kaplan-Meier survival curves for two pre-defined subject groups
    with a log-rank test p-value annotation.

    Useful for biological sanity checks (e.g. APOE4 carriers vs. non-carriers,
    amyloid-positive vs. amyloid-negative) and for understanding which patient
    subgroups drive model predictions.  Groups with fewer than 5 subjects are
    skipped silently to avoid unstable KM estimates.

    Args:
        y_event (np.ndarray): Binary event indicators, shape ``(n,)``.
            ``1`` = event observed; ``0`` = censored.
        y_duration (np.ndarray): Time to event or censoring in years, shape ``(n,)``.
        group_mask (np.ndarray[bool]): Boolean mask of length ``n``.
            ``True`` selects group 1; ``False`` selects group 0.
        group_labels (tuple[str, str]): Display names for
            ``(group0, group1)`` in the legend.
        title (str): Plot title string.
        fname_stem (str): Output filename stem (no extension, no directory).
            For example ``'km_apoe4'`` saves to ``FIG_DIR/km_apoe4_<cohort>.png``.
        cohort (str): Cohort label appended to the filename to disambiguate
            MCI and CN cohort outputs.  Default ``''``.

    Returns:
        None.  Saves figure to ``FIG_DIR/<fname_stem>_<cohort>.png`` and
        displays it inline.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)
    kmf = KaplanMeierFitter()

    for mask, label, col in [
        (~group_mask, group_labels[0], PALETTE[0]),
        ( group_mask, group_labels[1], PALETTE[1]),
    ]:
        if mask.sum() < 5:
            continue
        kmf.fit(y_duration[mask], event_observed=y_event[mask],
                label=f'{label} (n={mask.sum()})')
        kmf.plot_survival_function(ax=ax, color=col, ci_show=True, ci_alpha=0.2,
                                   linewidth=2.2)

    try:
        lr = logrank_test(
            y_duration[group_mask], y_duration[~group_mask],
            event_observed_A=y_event[group_mask],
            event_observed_B=y_event[~group_mask]
        )
        p_str = f'p = {lr.p_value:.4f}' if lr.p_value >= 0.0001 else 'p < 0.0001'
        ax.text(0.97, 0.97, f'Log-rank {p_str}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=10, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc', alpha=0.8))
    except Exception:
        pass

    ax.set(xlabel='Years from Baseline', ylabel='P(No Event)',
           title=title, ylim=(0, 1.02))
    ax.axhline(0.5, color='gray', ls=':', alpha=0.6)
    ax.legend(fontsize=10, framealpha=0.9)
    plt.tight_layout()

    cohort_clean = cohort.replace('>', '')
    fname = FIG_DIR / f'{fname_stem}_{cohort_clean}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {fname}')


# ═══════════════════════════════════════════════════════════════════════════════
# Individual survival curves
# ═══════════════════════════════════════════════════════════════════════════════

def plot_individual_survival_curves(
    curves: List[pd.Series],
    duration: float,
    event: int,
    labels: Optional[List[str]] = None,
    title: str = "Survival Curves"
) -> None:
    """
    Plot multiple predicted survival curves for a **single subject** with
    a vertical line at the observed exit time and an event-status marker.

    Designed for side-by-side model comparison: pass one ``pd.Series`` per
    model (e.g. CoxPH, GBSA, DeepSurv) to compare how differently each model
    characterises the same patient's trajectory.

    Args:
        curves (List[pd.Series]): Each series contains predicted survival
            probabilities ``S(t|x)`` indexed by time in years.  All series
            should cover a similar time range, but exact time grids need not
            match.
        duration (float): The subject's observed exit time (event or censoring
            time) in years.  A vertical dashed line is drawn at this point.
        event (int): Outcome at ``duration``.
            - ``1`` = event occurred (diamond marker, red fill)
            - ``0`` = censored (circle marker, open fill)
        labels (Optional[List[str]]): Display names for each curve, one per
            element of ``curves``.  Defaults to ``['Curve 1', 'Curve 2', ...]``.
        title (str): Plot title.  Default ``'Survival Curves'``.

    Returns:
        None.  Displays the figure inline (does not save to disk; call
        ``plt.savefig`` manually if persistence is needed).

    Raises:
        ValueError: If ``event`` is not 0 or 1, or if the number of labels
            does not match the number of curves.
    """
    if event not in (0, 1):
        raise ValueError("`event` must be 0 (censored) or 1 (event).")
    if labels is None:
        labels = [f"Curve {i+1}" for i in range(len(curves))]
    elif len(labels) != len(curves):
        raise ValueError("labels length must match curves length.")

    GRAY = "#888780"
    RED  = "#E24B4A"
    color_cycler = itertools.cycle(PALETTE)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    curve_colors = []
    for series, label in zip(curves, labels):
        color = next(color_cycler)
        curve_colors.append(color)
        ax.step(series.index, series.values, where="post",
                color=color, linewidth=2.2, label=label)

    ax.axvline(duration, color=GRAY, linewidth=1.2, linestyle="--", alpha=0.8,
               label=f"t = {duration:.2f}")

    def interp(series, t):
        times = series.index.to_numpy(dtype=float)
        probs = series.values.astype(float)
        if t <= times[0]:   return float(probs[0])
        if t >= times[-1]:  return float(probs[-1])
        return float(np.interp(t, times, probs))

    event_label = "Event" if event == 1 else "Censored"
    marker = "D" if event == 1 else "o"
    mfc    = RED if event == 1 else "none"

    for series, label, color in zip(curves, labels, curve_colors):
        surv_val = interp(series, duration)
        ax.plot(duration, surv_val, marker=marker, markersize=10,
                markerfacecolor=mfc, markeredgecolor=color,
                markeredgewidth=2, zorder=5,
                label=f"{label} S({duration:.2f})={surv_val:.3f} [{event_label}]")

    ax.set_xlabel("Time (years)", fontsize=12, color=GRAY)
    ax.set_ylabel("Survival probability S(t)", fontsize=12, color=GRAY)
    ax.set_title(title, fontsize=14, fontweight="normal", pad=14)
    ax.set_ylim(-0.02, 1.08)
    ax.tick_params(colors=GRAY, labelsize=10)
    ax.legend(frameon=True, framealpha=0.9, fontsize=8,
              edgecolor="#D3D1C7",
              bbox_to_anchor=(0.5, -0.22), loc='upper center',
              borderaxespad=0, ncol=3)
    fig.tight_layout(pad=3.0)
    plt.show()


def plot_median_survival_by_group(surv_df, y_event, y_duration, group_mask,
                                   group_labels, title, fname_stem, cohort=''):
    """
    Side-by-side plot of mean predicted survival curves and empirical KM curves
    for two subject subgroups, used to assess both discrimination and calibration.

    The **left panel** shows the average ``S(t|x)`` predicted by the model for
    each group (± IQR band).  The **right panel** shows the empirical KM estimate
    for the same groups.  When these two panels agree, the model is well-calibrated;
    when predicted and observed group separations differ in magnitude, the model
    either exaggerates or understates the true risk difference.

    Args:
        surv_df (pd.DataFrame): Predicted survival matrix, shape
            ``(n_timepoints, n_subjects)``.  Index = time points in years;
            columns = subject indices in the same order as ``y_event``.
        y_event (np.ndarray): Binary event indicators, shape ``(n,)``.
        y_duration (np.ndarray): Time to event or censoring in years, shape ``(n,)``.
        group_mask (np.ndarray[bool]): Boolean mask of length ``n``.
            ``True`` = group 1 (e.g. converters); ``False`` = group 0 (stable).
        group_labels (tuple[str, str]): Display names for ``(group0, group1)``.
        title (str): Figure super-title.
        fname_stem (str): Output filename stem (no extension).
            Saved to ``FIG_DIR/<fname_stem>_median_surv_<cohort>.png``.
        cohort (str): Cohort label for filename disambiguation.  Default ``''``.

    Returns:
        None.  Saves figure to ``FIG_DIR`` and displays it inline.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(BG)
    fig.suptitle(title, fontsize=14, y=1.01)

    # Left: average predicted curves
    ax = axes[0]
    _style_ax(ax)
    times = surv_df.index.to_numpy()
    surv_arr = surv_df.values

    for mask, label, col in [
        (~group_mask, group_labels[0], PALETTE[0]),
        ( group_mask, group_labels[1], PALETTE[1]),
    ]:
        if mask.sum() < 2:
            continue
        mean_curve = surv_arr[:, mask].mean(axis=1)
        q25  = np.percentile(surv_arr[:, mask], 25, axis=1)
        q75  = np.percentile(surv_arr[:, mask], 75, axis=1)
        ax.step(times, mean_curve, where='post', color=col, linewidth=2.2,
                label=f'{label} (n={mask.sum()})')
        ax.fill_between(times, q25, q75, step='post', color=col, alpha=0.15)

    ax.set(xlabel='Years', ylabel='Mean S(t|x)',
           title='Mean Predicted Survival ± IQR', ylim=(0, 1.02))
    ax.legend(fontsize=9)

    # Right: KM curves
    ax2 = axes[1]
    _style_ax(ax2)
    kmf = KaplanMeierFitter()
    for mask, label, col in [
        (~group_mask, group_labels[0], PALETTE[0]),
        ( group_mask, group_labels[1], PALETTE[1]),
    ]:
        if mask.sum() < 5:
            continue
        kmf.fit(y_duration[mask], event_observed=y_event[mask],
                label=f'{label} (n={mask.sum()})')
        kmf.plot_survival_function(ax=ax2, color=col, ci_show=True, ci_alpha=0.2,
                                   linewidth=2.2)
    ax2.set(xlabel='Years', ylabel='KM P(Event-Free)',
            title='Empirical KM Curves', ylim=(0, 1.02))
    ax2.legend(fontsize=9)

    plt.tight_layout()
    cohort_clean = cohort.replace('>', '')
    fname = FIG_DIR / f'{fname_stem}_median_surv_{cohort_clean}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {fname}')


# ═══════════════════════════════════════════════════════════════════════════════
# Feature importance
# ═══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(importance_series, model_name, cohort, top_n=20,
                             perm_df=None):
    """
    Horizontal bar chart of feature importances, optionally overlaid with
    permutation importance error bars.

    Args:
        importance_series (pd.Series): Native importance values indexed by
            feature name, sorted descending (e.g. from GBSA or Cox PH coefs).
        model_name (str): Used in title and filename.
        cohort (str): Cohort label.
        top_n (int): Number of top features to display. Default 20.
        perm_df (pd.DataFrame | None): Optional permutation importance from
            permutation_importance_survival(), with columns
            ['feature', 'importance_mean', 'importance_std'].
            If provided, mean ± std error bars are overlaid.

    Saves figure to FIG_DIR as 'feat_imp_<model>_<cohort>.png'.
    """
    top = importance_series.head(top_n)
    cohort_clean = cohort.replace('>', '')

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.38)))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    colors = [PALETTE[0]] * len(top)
    ax.barh(top.index[::-1], top.values[::-1], color=colors, height=0.65, zorder=2)

    # Overlay permutation importance if provided
    if perm_df is not None:
        perm_sub = perm_df[perm_df['feature'].isin(top.index)].set_index('feature')
        # Normalize perm to same scale as native importance for overlay
        perm_vals = perm_sub.reindex(top.index)['importance_mean']
        perm_stds = perm_sub.reindex(top.index)['importance_std']
        if perm_vals.notna().any():
            scale = top.max() / (perm_vals.max() + 1e-9)
            ax.errorbar(
                perm_vals[::-1].fillna(0).values * scale,
                range(len(top)),
                xerr=perm_stds[::-1].fillna(0).values * scale,
                fmt='D', color=PALETTE[1], markersize=5, linewidth=1.5,
                label='Permutation imp. (scaled)', zorder=3
            )
            ax.legend(fontsize=9)

    ax.set(xlabel='Importance', title=f'Feature Importance: {model_name} [{cohort}]')
    ax.axvline(0, color='#aaa', linewidth=0.8)
    plt.tight_layout()
    fname = FIG_DIR / f'feat_imp_{model_name.replace(" ","_")}_{cohort_clean}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {fname}')


def plot_cox_coefficients(model, scaler, feature_names, model_name, cohort, top_n=20):
    """
    Signed coefficient plot for Cox PH — shows direction (protective vs. hazardous)
    and magnitude. Positive = increases hazard; negative = reduces hazard.

    Args:
        model (CoxPHFitter): Fitted lifelines CoxPHFitter.
        scaler (StandardScaler): Scaler used to standardise inputs.
        feature_names (list[str]): Original feature names.
        model_name (str): Used in title.
        cohort, top_n: As in plot_feature_importance.

    Saves to FIG_DIR as 'cox_coef_<model>_<cohort>.png'.
    """
    coefs = model.params_.copy()
    coefs.index = feature_names[:len(coefs)]
    top_abs = coefs.abs().nlargest(top_n)
    top_coefs = coefs[top_abs.index].sort_values()

    cohort_clean = cohort.replace('>', '')
    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.38)))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    bar_colors = [PALETTE[1] if v > 0 else PALETTE[0] for v in top_coefs.values]
    ax.barh(top_coefs.index, top_coefs.values, color=bar_colors, height=0.65, zorder=2)
    ax.axvline(0, color='black', linewidth=1.0)
    ax.set(xlabel='Coefficient (log-hazard ratio)',
           title=f'Cox PH Coefficients: {model_name} [{cohort}]')

    pos_patch = mpatches.Patch(color=PALETTE[1], label='Increases hazard')
    neg_patch = mpatches.Patch(color=PALETTE[0], label='Reduces hazard')
    ax.legend(handles=[pos_patch, neg_patch], fontsize=9)
    plt.tight_layout()

    fname = FIG_DIR / f'cox_coef_{model_name.replace(" ","_")}_{cohort_clean}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {fname}')


def plot_permutation_importance(perm_df, model_name, cohort, top_n=20):
    """
    Horizontal bar chart of permutation feature importance with ± 1 SD error bars.

    Each bar shows the mean drop in Harrell's C-index when a feature's values
    are randomly shuffled, breaking its relationship with the outcome.  The
    error bars (± 1 SD across repeats) indicate stability — wide bars mean the
    importance estimate is noisy and should be interpreted with caution.

    Features with **negative importance** (bar extends left of zero) are
    features whose removal accidentally *improves* ranking, indicating they
    add noise to the model.  Consider dropping such features in a refined model.

    Args:
        perm_df (pd.DataFrame): Output of ``permutation_importance_survival()``.
            Required columns: ``'feature'``, ``'importance_mean'``,
            ``'importance_std'``.  Must be sorted descending by
            ``'importance_mean'`` (the function does this automatically).
        model_name (str): Used in the plot title and output filename.
        cohort (str): Cohort label (e.g. ``'MCI->Dementia'``).
        top_n (int): Number of top features to display.  Default ``20``.

    Returns:
        None.  Saves figure to ``FIG_DIR/perm_imp_<model>_<cohort>.png``
        and displays it inline.
    """
    top = perm_df.head(top_n).copy()
    cohort_clean = cohort.replace('>', '')

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.38)))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    ax.barh(top['feature'][::-1], top['importance_mean'][::-1],
            xerr=top['importance_std'][::-1],
            color=PALETTE[2], height=0.65, zorder=2,
            error_kw={'elinewidth': 1.5, 'capsize': 3, 'ecolor': '#555'})
    ax.axvline(0, color='#aaa', linewidth=0.8)
    ax.set(xlabel='Mean C-index drop when feature is shuffled',
           title=f'Permutation Importance: {model_name} [{cohort}]')
    plt.tight_layout()

    fname = FIG_DIR / f'perm_imp_{model_name.replace(" ","_")}_{cohort_clean}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {fname}')


# ═══════════════════════════════════════════════════════════════════════════════
# Overfitting diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def plot_learning_curve_deepsurv(loss_history, model_name, cohort):
    """
    Plot per-epoch training and validation loss for the final DeepSurv refit.

    Loss is the negative partial log-likelihood (lower = better fit to the
    Cox model on that split).  The plot is the primary diagnostic for
    epoch-level overfitting in DeepSurv.

    A **healthy curve** shows:
    - Both train and val loss decreasing in early epochs.
    - Val loss levelling off or improving more slowly than train loss.
    - Early stopping fires near the val loss minimum.

    An **overfitting signature**:
    - Train loss continues decreasing while val loss starts *increasing*.
    - The gap between the two curves widens monotonically.

    If ``loss_history`` is empty (both lists are ``[]``), the pycox training
    log format was incompatible with extraction — a warning is printed
    and the function returns early without plotting.

    Args:
        loss_history (dict): Dictionary with two keys:

            - ``'train'`` (list[float]): Per-epoch negative partial
              log-likelihood on the training split.
            - ``'val'``   (list[float]): Per-epoch loss on the validation split.

            Both lists are returned by ``run_deepsurv``.  If empty, no plot
            is produced.
        model_name (str): Used in the plot title and output filename.
        cohort (str): Cohort label (e.g. ``'MCI->Dementia'``).

    Returns:
        None.  Saves figure to ``FIG_DIR/deepsurv_loss_<model>_<cohort>.png``
        and displays it inline.  If loss history is unavailable, prints a
        warning and returns without saving.
    """
    if not loss_history.get('train') and not loss_history.get('val'):
        print('  No loss history available for DeepSurv.')
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    epochs = range(1, len(loss_history['train']) + 1)
    if loss_history['train']:
        ax.plot(epochs, loss_history['train'], color=PALETTE[0], linewidth=2,
                label='Train loss')
    if loss_history['val']:
        val_epochs = range(1, len(loss_history['val']) + 1)
        ax.plot(val_epochs, loss_history['val'], color=PALETTE[1], linewidth=2,
                linestyle='--', label='Val loss')

    ax.set(xlabel='Epoch', ylabel='Neg. Partial Log-Likelihood',
           title=f'DeepSurv Learning Curves: {model_name} [{cohort}]')
    ax.legend(fontsize=10)
    plt.tight_layout()

    cohort_clean = cohort.replace('>', '')
    fname = FIG_DIR / f'deepsurv_loss_{model_name.replace(" ","_")}_{cohort_clean}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {fname}')


def plot_overfitting_summary(results_dict, cohort):
    """
    Grouped bar chart comparing OOF (cross-validation) vs. in-sample C-index
    and holdout test C-index for all models in one cohort.

    A large OOF-to-train gap (blue bar much shorter than orange) signals overfitting.
    The green bar (holdout) is the honest estimate.

    Args:
        results_dict (dict): Keys are model names. Values are dicts with keys:
            'oof_c'     : float — OOF / CV concordance.
            'train_c'   : float — In-sample concordance.
            'test_c'    : float — Holdout test concordance (optional).
        cohort (str): Cohort label, used in title and filename.

    Saves to FIG_DIR as 'overfitting_summary_<cohort>.png'.
    """
    model_names = list(results_dict.keys())
    oof_vals   = [results_dict[m].get('oof_c',   np.nan) for m in model_names]
    train_vals = [results_dict[m].get('train_c', np.nan) for m in model_names]
    test_vals  = [results_dict[m].get('test_c',  np.nan) for m in model_names]

    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    ax.bar(x - width, oof_vals,   width, label='OOF / CV C-index',     color=PALETTE[0], zorder=2)
    ax.bar(x,         train_vals, width, label='In-sample C-index',     color=PALETTE[1], zorder=2)
    ax.bar(x + width, test_vals,  width, label='Holdout test C-index',  color=PALETTE[2], zorder=2)

    ax.axhline(0.5, color='gray', ls=':', linewidth=1, label='Random (C=0.5)')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=10)
    ax.set(ylabel='C-index', ylim=(0.45, 1.01),
           title=f'Overfitting Summary: OOF vs. Train vs. Test [{cohort}]')
    ax.legend(fontsize=9, framealpha=0.9)

    for bars_x, vals in [(x - width, oof_vals), (x, train_vals), (x + width, test_vals)]:
        for bx, v in zip(bars_x, vals):
            if not np.isnan(v):
                ax.text(bx, v + 0.005, f'{v:.3f}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold')

    plt.tight_layout()
    cohort_clean = cohort.replace('>', '')
    fname = FIG_DIR / f'overfitting_summary_{cohort_clean}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {fname}')


# ═══════════════════════════════════════════════════════════════════════════════
# Bootstrap CI
# ═══════════════════════════════════════════════════════════════════════════════

def plot_bootstrap_ci(boot_results_dict, cohort, metric_label='C-index'):
    """
    Violin + box plot showing the bootstrap distribution of the C-index for
    each model, with the 95% CI band marked.

    Args:
        boot_results_dict (dict): Keys are model names. Values are dicts with:
            'point' : float  — Point estimate on holdout.
            'lower' : float  — Lower 95% CI bound.
            'upper' : float  — Upper 95% CI bound.
            'boot'  : np.ndarray — Full bootstrap distribution.
        cohort (str): Cohort label.
        metric_label (str): Y-axis label. Default 'C-index'.

    Saves to FIG_DIR as 'bootstrap_ci_<cohort>.png'.
    """
    model_names = list(boot_results_dict.keys())
    boot_arrays = [boot_results_dict[m]['boot'] for m in model_names]

    fig, ax = plt.subplots(figsize=(max(7, len(model_names) * 2), 6))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    parts = ax.violinplot(boot_arrays, positions=range(len(model_names)),
                          showmedians=True, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(PALETTE[i % len(PALETTE)])
        pc.set_alpha(0.55)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)

    # Overlay 95% CI as error bars
    for i, m in enumerate(model_names):
        res = boot_results_dict[m]
        ax.errorbar(i, res['point'], yerr=[[res['point'] - res['lower']],
                                            [res['upper'] - res['point']]],
                    fmt='D', color='black', markersize=7, linewidth=2, zorder=5,
                    label='Point est. ± 95% CI' if i == 0 else '')

    ax.axhline(0.5, color='gray', ls=':', linewidth=1)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=10)
    ax.set(ylabel=metric_label,
           title=f'Bootstrap 95% CI: Holdout {metric_label} [{cohort}]')
    ax.legend(fontsize=9)
    plt.tight_layout()

    cohort_clean = cohort.replace('>', '')
    fname = FIG_DIR / f'bootstrap_ci_{cohort_clean}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {fname}')


def plot_bootstrap_distribution(boot_dict, model_name, cohort):
    """
    Histogram of the bootstrap C-index distribution for a single model with
    the 95 % CI region shaded and the point estimate marked.

    The bootstrap distribution reflects how stable the C-index estimate is:
    a **narrow, symmetric bell curve** means the estimate is reliable and
    would change little if a different random holdout sample had been drawn.
    A **wide or skewed distribution** means the estimate is sensitive to
    which subjects end up in the test set — often caused by a small test
    set or a low event rate.

    Args:
        boot_dict (dict): Output of ``bootstrap_cindex_harrell`` or
            ``bootstrap_cindex_td`` for one model.  Required keys:

            - ``'point'`` (float): C-index on the original holdout data.
            - ``'lower'`` (float): Lower 95 % CI percentile.
            - ``'upper'`` (float): Upper 95 % CI percentile.
            - ``'boot'``  (np.ndarray): All bootstrap C-index values.

        model_name (str): Used in the plot title and output filename.
        cohort (str): Cohort label (e.g. ``'MCI->Dementia'``).

    Returns:
        None.  Saves figure to ``FIG_DIR/boot_dist_<model>_<cohort>.png``
        and displays it inline.
    """
    boot = boot_dict['boot']
    point = boot_dict['point']
    lo, hi = boot_dict['lower'], boot_dict['upper']

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    ax.hist(boot, bins=40, color=PALETTE[0], alpha=0.75, edgecolor='white',
            zorder=2, density=True)
    ax.axvline(point, color=PALETTE[1], linewidth=2.5, label=f'Point: {point:.4f}')
    ax.axvline(lo, color=PALETTE[1], linewidth=1.5, linestyle='--',
               label=f'95% CI [{lo:.4f}, {hi:.4f}]')
    ax.axvline(hi, color=PALETTE[1], linewidth=1.5, linestyle='--')
    ax.axvspan(lo, hi, alpha=0.12, color=PALETTE[1])

    ax.set(xlabel='C-index', ylabel='Density',
           title=f'Bootstrap Distribution: {model_name} [{cohort}]')
    ax.legend(fontsize=10)
    plt.tight_layout()

    cohort_clean = cohort.replace('>', '')
    fname = FIG_DIR / f'boot_dist_{model_name.replace(" ","_")}_{cohort_clean}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {fname}')


# ═══════════════════════════════════════════════════════════════════════════════
# ROC at fixed horizons
# ═══════════════════════════════════════════════════════════════════════════════

def plot_roc_horizons(surv_df, y_event, y_duration, model_name, cohort,
                       horizons=HORIZONS):
    """
    ROC curves at fixed prediction horizons (e.g. 3-year and 5-year) using
    cross-validated binary labels. Area under each curve is annotated.

    Subjects with uncertain outcomes at the horizon (censored before horizon)
    are excluded from the AUC calculation.

    Args:
        surv_df (pd.DataFrame): Survival matrix (n_times × n_subjects).
        y_event, y_duration (np.ndarray): Survival labels.
        model_name (str): Used in title and filename.
        cohort (str): Cohort label.
        horizons (list[int]): Time horizons in years. Default HORIZONS.

    Saves to FIG_DIR as 'roc_horizons_<model>_<cohort>.png'.
    """
    from sklearn.metrics import roc_curve, auc

    times = surv_df.index.to_numpy()
    surv_arr = surv_df.values

    fig, axes = plt.subplots(1, len(horizons), figsize=(6 * len(horizons), 5),
                              sharey=True)
    fig.patch.set_facecolor(BG)
    if len(horizons) == 1:
        axes = [axes]

    for ax, h in zip(axes, horizons):
        _style_ax(ax)
        y_bin, include = binary_horizon_dataset(y_event, y_duration, h)
        if y_bin.sum() < 10:
            ax.text(0.5, 0.5, f'Too few events\nat {h}yr horizon',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{h}-Year Horizon')
            continue

        # Risk score = 1 - S(t=horizon|x) for included subjects
        t_idx = np.searchsorted(times, h)
        t_idx = min(t_idx, len(times) - 1)
        risk_h = 1.0 - surv_arr[t_idx, include]

        fpr, tpr, _ = roc_curve(y_bin, risk_h)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=PALETTE[0], linewidth=2.2,
                label=f'AUC = {roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
        ax.fill_between(fpr, tpr, alpha=0.10, color=PALETTE[0])
        ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate',
               title=f'{h}-Year ROC [{model_name}]', xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(fontsize=10, loc='lower right')

    fig.suptitle(f'Fixed-Horizon ROC: {cohort}', fontsize=13, y=1.02)
    plt.tight_layout()
    cohort_clean = cohort.replace('>', '')
    fname = FIG_DIR / f'roc_horizons_{model_name.replace(" ","_")}_{cohort_clean}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {fname}')


# ═══════════════════════════════════════════════════════════════════════════════
# Brier score
# ═══════════════════════════════════════════════════════════════════════════════

def plot_brier_score(surv_df, y_event, y_duration, model_name, cohort,
                      t_max=None, n_times=50):
    """
    Time-dependent Brier score curve using the IPCW estimator (Graf et al.).

    The integrated Brier score (IBS) across all time points is also computed
    and annotated. Lower is better (0 = perfect, 0.25 = random).

    Args:
        surv_df (pd.DataFrame): Survival matrix (n_times × n_subjects) with
            S(t|x) predictions. Rows are time points, columns are patients.
        y_event, y_duration (np.ndarray): Survival labels.
        model_name (str): Used in title and filename.
        cohort (str): Cohort label.
        t_max (float | None): Maximum time for Brier score evaluation.
            Defaults to 80th percentile of observed times to avoid instability
            in sparse tail regions.
        n_times (int): Number of evaluation time points. Default 50.

    Saves to FIG_DIR as 'brier_<model>_<cohort>.png'.
    """
    from lifelines import KaplanMeierFitter as _KMF

    times   = surv_df.index.to_numpy(dtype=np.float64)
    surv_arr = surv_df.values.astype(np.float64)
    n = len(y_event)

    if t_max is None:
        t_max = np.percentile(y_duration, 80)

    eval_times = np.linspace(times.min() + 1e-4, t_max, n_times)

    # IPCW weights: fit KM on censoring distribution (event = 1 - observed event)
    kmf_cens = _KMF()
    kmf_cens.fit(y_duration, event_observed=(1 - y_event))

    brier_scores = []
    for t in eval_times:
        t_idx = np.clip(np.searchsorted(times, t), 0, len(times) - 1)
        s_t = surv_arr[t_idx, :]

        # IPCW weights at event times
        g_t_i = np.clip(kmf_cens.survival_function_at_times(
            np.minimum(y_duration, t)).values.ravel(), 1e-4, None)
        g_t   = np.clip(kmf_cens.survival_function_at_times([t]).values[0], 1e-4, None)

        bs = (
            np.sum(((1 - s_t) ** 2) * (y_duration <= t) * y_event / g_t_i)
            + np.sum((s_t ** 2) * (y_duration > t) / g_t)
        ) / n
        brier_scores.append(bs)

    brier_scores = np.array(brier_scores)
    # Integrated Brier Score via trapezoidal rule (normalised by time range)
    ibs = np.trapz(brier_scores, eval_times) / (eval_times[-1] - eval_times[0])

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    ax.plot(eval_times, brier_scores, color=PALETTE[0], linewidth=2.2,
            label=f'Brier score (IBS={ibs:.4f})')
    ax.axhline(0.25, color='gray', ls='--', linewidth=1, label='Reference (0.25)')
    ax.fill_between(eval_times, brier_scores, alpha=0.15, color=PALETTE[0])
    ax.set(xlabel='Time (years)', ylabel='Brier Score',
           title=f'Time-Dependent Brier Score: {model_name} [{cohort}]',
           ylim=(0, min(0.30, brier_scores.max() * 1.3)))
    ax.legend(fontsize=10)
    plt.tight_layout()

    cohort_clean = cohort.replace('>', '')
    fname = FIG_DIR / f'brier_{model_name.replace(" ","_")}_{cohort_clean}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {fname}  |  IBS = {ibs:.4f}')
    return ibs


# ═══════════════════════════════════════════════════════════════════════════════
# Calibration at fixed horizons
# ═══════════════════════════════════════════════════════════════════════════════

def plot_calibration_horizon(surv_df, y_event, y_duration, model_name, cohort,
                              horizon=3, n_bins=5):
    """
    Calibration plot at a fixed horizon: predicted vs. observed event probability,
    binned by predicted risk (higher bins = higher predicted risk).

    Observed event rates are estimated via KM within each bin. A perfectly
    calibrated model falls on the diagonal.

    Args:
        surv_df (pd.DataFrame): Survival matrix (n_times × n_subjects).
        y_event, y_duration (np.ndarray): Survival labels.
        model_name (str): Used in title and filename.
        cohort (str): Cohort label.
        horizon (int): Time horizon in years. Default 3.
        n_bins (int): Number of risk bins. Default 5.

    Saves to FIG_DIR as 'calibration_<horizon>yr_<model>_<cohort>.png'.
    """
    times = surv_df.index.to_numpy(dtype=np.float64)
    surv_arr = surv_df.values.astype(np.float64)

    t_idx = np.clip(np.searchsorted(times, horizon), 0, len(times) - 1)
    pred_event_prob = 1.0 - surv_arr[t_idx, :]  # P(event by horizon | x)

    # Bin by predicted probability
    try:
        bins = pd.qcut(pred_event_prob, q=n_bins, labels=False, duplicates='drop')
    except ValueError:
        print(f'  Calibration: not enough distinct predicted values for {n_bins} bins.')
        return

    pred_means, obs_means, obs_cis_lo, obs_cis_hi = [], [], [], []
    kmf = KaplanMeierFitter()

    for b in range(bins.max() + 1):
        mask = bins == b
        if mask.sum() < 5:
            continue
        pred_means.append(pred_event_prob[mask].mean())
        # Observed: 1 - KM survival at horizon
        kmf.fit(y_duration[mask], event_observed=y_event[mask])
        km_vals = kmf.survival_function_at_times([horizon]).values
        km_ci   = kmf.confidence_interval_survival_function_at_times([horizon])
        obs_means.append(float(1 - km_vals[0]))
        obs_cis_lo.append(float(1 - km_ci.iloc[0, 1]))  # upper CI → lower event prob
        obs_cis_hi.append(float(1 - km_ci.iloc[0, 0]))  # lower CI → upper event prob

    pred_means  = np.array(pred_means)
    obs_means   = np.array(obs_means)
    obs_cis_lo  = np.array(obs_cis_lo)
    obs_cis_hi  = np.array(obs_cis_hi)

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1,
            label='Perfect calibration')
    ax.errorbar(pred_means, obs_means,
                yerr=[obs_means - obs_cis_lo, obs_cis_hi - obs_means],
                fmt='o', color=PALETTE[0], markersize=8, linewidth=2,
                label='Predicted vs. KM-observed', capsize=4, zorder=3)
    ax.plot(pred_means, obs_means, color=PALETTE[0], linewidth=1.5, alpha=0.5)

    ax.set(xlabel=f'Mean Predicted P(Event by {horizon}yr)',
           ylabel=f'Observed P(Event by {horizon}yr) via KM',
           title=f'Calibration at {horizon}yr: {model_name} [{cohort}]',
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=10)
    plt.tight_layout()

    cohort_clean = cohort.replace('>', '')
    fname = FIG_DIR / (f'calibration_{horizon}yr_{model_name.replace(" ","_")}'
                        f'_{cohort_clean}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {fname}')


# ═══════════════════════════════════════════════════════════════════════════════
# Model comparison summary
# ═══════════════════════════════════════════════════════════════════════════════

def plot_model_comparison(boot_results_dict, cohort, metric_label='Holdout C-index (IPCW)'):
    """
    Horizontal bar chart comparing all models on the holdout test set with
    95% bootstrap confidence intervals.

    Args:
        boot_results_dict (dict): Same format as in plot_bootstrap_ci.
        cohort (str): Cohort label.
        metric_label (str): X-axis label.

    Saves to FIG_DIR as 'model_comparison_<cohort>.png'.
    """
    model_names  = list(boot_results_dict.keys())
    points = [boot_results_dict[m]['point'] for m in model_names]
    lowers = [boot_results_dict[m]['lower'] for m in model_names]
    uppers = [boot_results_dict[m]['upper'] for m in model_names]

    order = np.argsort(points)
    model_names = [model_names[i] for i in order]
    points  = [points[i]  for i in order]
    lowers  = [lowers[i]  for i in order]
    uppers  = [uppers[i]  for i in order]
    xerr_lo = [p - lo for p, lo in zip(points, lowers)]
    xerr_hi = [hi - p  for p, hi in zip(points, uppers)]

    fig, ax = plt.subplots(figsize=(9, max(4, len(model_names) * 0.9)))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(model_names))]
    ax.barh(model_names, points, xerr=[xerr_lo, xerr_hi], color=colors,
            height=0.55, zorder=2, capsize=4,
            error_kw={'elinewidth': 2, 'capthick': 2, 'ecolor': '#333'})
    ax.axvline(0.5, color='gray', ls='--', linewidth=1, label='Random (C=0.5)')

    for i, (p, lo, hi) in enumerate(zip(points, lowers, uppers)):
        ax.text(hi + 0.003, i, f'{p:.3f} [{lo:.3f}–{hi:.3f}]',
                va='center', fontsize=9)

    ax.set(xlabel=metric_label,
           title=f'Model Comparison (95% Bootstrap CI): {cohort}')
    ax.legend(fontsize=9)
    plt.tight_layout()

    cohort_clean = cohort.replace('>', '')
    fname = FIG_DIR / f'model_comparison_{cohort_clean}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {fname}')


# ═══════════════════════════════════════════════════════════════════════════════
# Optuna HPO visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def plot_optuna_history(study, model_name, cohort):
    """
    Scatter plot of all Optuna trial values over time with the running best
    value overlaid as a step curve.

    Args:
        study (optuna.Study): Completed Optuna study.
        model_name (str): Used in title and filename.
        cohort (str): Cohort label.

    Saves to FIG_DIR as 'optuna_history_<model>_<cohort>.png'.
    """
    trial_vals = [t.value for t in study.trials if t.value is not None]
    best_vals  = np.maximum.accumulate(trial_vals)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    ax.scatter(range(len(trial_vals)), trial_vals, color=PALETTE[0],
               alpha=0.5, s=20, zorder=2, label='Trial')
    ax.step(range(len(best_vals)), best_vals, color=PALETTE[1],
            linewidth=2, where='post', label='Best so far')
    ax.set(xlabel='Trial', ylabel='OOF C-index',
           title=f'Optuna HPO History: {model_name} [{cohort}]')
    ax.legend(fontsize=10)
    plt.tight_layout()

    cohort_clean = cohort.replace('>', '')
    fname = FIG_DIR / f'optuna_history_{model_name.replace(" ","_")}_{cohort_clean}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {fname}')


# ═══════════════════════════════════════════════════════════════════════════════
# Survival curve grid for a small set of subjects
# ═══════════════════════════════════════════════════════════════════════════════

def plot_survival_curve_grid(surv_df, y_event, y_duration, model_name, cohort,
                              n_subjects=12, seed=RANDOM_SEED):
    """
    Grid of individual predicted survival curves for a random sample of subjects,
    colour-coded by event status (event = red, censored = blue).

    Useful for a sanity check — converters should have steeper curves than stable
    subjects. Each panel shows the observed exit time as a vertical line.

    Args:
        surv_df (pd.DataFrame): Survival matrix (n_times × n_subjects).
        y_event, y_duration (np.ndarray): Survival labels.
        model_name (str): Title and filename.
        cohort (str): Cohort label.
        n_subjects (int): Number of subjects to sample. Default 12.
        seed (int): Random seed.

    Saves to FIG_DIR as 'surv_grid_<model>_<cohort>.png'.
    """
    rng = np.random.RandomState(seed)
    times   = surv_df.index.to_numpy()
    surv_arr = surv_df.values
    n = surv_arr.shape[1]
    n_subjects = min(n_subjects, n)
    idx = rng.choice(n, n_subjects, replace=False)

    ncols = 4
    nrows = int(np.ceil(n_subjects / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3),
                              sharey=True)
    fig.patch.set_facecolor(BG)
    axes = axes.ravel()

    for plot_i, subj_i in enumerate(idx):
        ax = axes[plot_i]
        _style_ax(ax)
        ev = int(y_event[subj_i])
        dur = y_duration[subj_i]
        col = PALETTE[1] if ev else PALETTE[0]
        label = 'Event' if ev else 'Censored'
        ax.step(times, surv_arr[:, subj_i], where='post', color=col, linewidth=1.8)
        ax.axvline(dur, color='gray', linestyle='--', linewidth=1, alpha=0.8)
        ax.set_title(f'Subject {subj_i}\n[{label}, t={dur:.1f}yr]',
                     fontsize=8, color=col)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(times[0], times[-1])
        ax.tick_params(labelsize=7)

    # Hide unused panels
    for ax in axes[n_subjects:]:
        ax.set_visible(False)

    event_patch  = mpatches.Patch(color=PALETTE[1], label='Event')
    censor_patch = mpatches.Patch(color=PALETTE[0], label='Censored')
    fig.legend(handles=[event_patch, censor_patch], loc='lower right', fontsize=10)
    fig.suptitle(f'Individual Survival Curves: {model_name} [{cohort}]',
                 fontsize=13, y=1.01)
    plt.tight_layout()

    cohort_clean = cohort.replace('>', '')
    fname = FIG_DIR / f'surv_grid_{model_name.replace(" ","_")}_{cohort_clean}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'  Saved: {fname}')


# ═══════════════════════════════════════════════════════════════════════════════
# Subject-level matrix (subject × time) builder (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════════

def build_subject_time_matrix(df_all, rids, time_grid, features, window=0.25):
    """
    Construct a 3-D tensor of longitudinal feature values aligned to a regular
    time grid for use with sequence models (Transformer, LSTM, etc.).

    For each subject × time-point, the nearest actual visit within ``±window``
    years is used.  NaN slots are forward-filled per subject, then any remaining
    NaN is replaced with the cross-subject median for that feature.

    Args:
        df_all (pd.DataFrame): Full longitudinal DataFrame with ``'RID'``
            and ``'Years_bl'`` columns plus all feature columns.
        rids (list): Subject IDs to include, in the desired row order.
        time_grid (np.ndarray): 1-D array of target time points in years.
        features (list[str]): Feature column names to extract.
        window (float): Maximum time gap (years) between a grid point and the
            nearest visit.  Default ``0.25`` (3 months).

    Returns:
        tuple:
            - **tensor** (np.ndarray, float32): Shape
              ``(n_subjects, n_timepoints, n_features)``.  NaN-free after
              forward-fill and median imputation.
            - **mask** (np.ndarray, bool): Shape ``(n_subjects, n_timepoints)``.
              ``True`` where a real observed value was used; ``False`` where
              the cell was imputed.
    """
    n_subj = len(rids)
    n_time = len(time_grid)
    n_feat = len(features)
    tensor = np.full((n_subj, n_time, n_feat), np.nan, dtype=np.float32)
    mask   = np.zeros((n_subj, n_time), dtype=bool)

    for s_idx, rid in enumerate(rids):
        subj = df_all[df_all['RID'] == rid].sort_values('Years_bl')
        for t_idx, t in enumerate(time_grid):
            diffs = np.abs(subj['Years_bl'].values - t)
            if diffs.min() <= window:
                nearest = subj.iloc[diffs.argmin()]
                for f_idx, feat in enumerate(features):
                    val = nearest[feat]
                    if not pd.isna(val):
                        tensor[s_idx, t_idx, f_idx] = val
                mask[s_idx, t_idx] = (diffs.min() <= window)

    for s_idx in range(n_subj):
        for f_idx in range(n_feat):
            arr = tensor[s_idx, :, f_idx]
            last = np.nan
            for t_idx in range(n_time):
                if not np.isnan(arr[t_idx]):
                    last = arr[t_idx]
                elif not np.isnan(last):
                    tensor[s_idx, t_idx, f_idx] = last
    for f_idx in range(n_feat):
        feat_median = np.nanmedian(tensor[:, :, f_idx])
        nan_mask = np.isnan(tensor[:, :, f_idx])
        tensor[:, :, f_idx][nan_mask] = feat_median

    return tensor, mask
