'''
this module includes functions related to stat analyses
'''

from collections import namedtuple
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import rich
import scikit_posthocs
from scipy import stats


def calc_dr(group1: pd.Series, group2: pd.Series) -> float:
    '''
    calculate the effect size (d_r) for two groups.

    d_r -- non-parametric (robust) variant of Cohen's d
    (Algina et al., 2005)
    '''
    diff = stats.trim_mean(group1, .2) - stats.trim_mean(group2, .2)
    n1, n2 = len(group1), len(group2)
    var1 = stats.mstats.winsorize(group1, .2).var()
    var2 = stats.mstats.winsorize(group2, .2).var()
    pooled_var = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return .642 * diff / pooled_var


def get_optimal_sample_size(*data: list,
                            stat_test: Callable = stats.kruskal,
                            effect_size_estimator: Callable = calc_dr,
                            alpha: float = .05,
                            desired_power: float = .8,
                            repetitions: int = 1000,
                            sample_size_min: int = 20,
                            max_step: int = 20,
                            ) -> Tuple[int, float, List[float]]:
    '''
    Use a subsampling approach to determine the optimal sample size.

    1. Determine the effect size (e.g., using d_r).
    2. Choose the significance level (alpha).
    3. Simulate the alternative hypothesis by adding the effect size
       to the control group (h0).
    4. Sample the simulated groups with a sample size of sample_size
       and perform the specified statistical test.
    5. Calculate the statistical power of the test (i.e., the proportion
       of times the null hypothesis was rejected at the specified
       significance level).
    6. Adjust the sample size until the desired statistical power is
       reached.

    returns tuple of sample size, power and effect sizes between groups
    '''
    max_sample_size = min((len(i) for i in data))

    # simulate groups
    h0 = data[0]
    effect_sizes = [effect_size_estimator(h0, gr) for gr in data[1:]]
    simulated_groups = [h0] + [h0 + es for es in effect_sizes]

    # obtain sample size
    sample_size = sample_size_min
    power = 0

    step = 0
    prev_sample_size = 0
    while not desired_power <= power < desired_power + .05:
        step += 1
        try:
            power = sum((stat_test(
                *[i.sample(sample_size)
                  for i in simulated_groups]).pvalue < alpha
                for _ in range(repetitions))) / repetitions
        except ValueError:
            sample_size += 1
            prev_sample_size = sample_size
            continue

        if power == 0:
            power = .00001

        sample_size += round(
            sample_size * desired_power / power) - sample_size

        if sample_size > max_sample_size or sample_size < sample_size_min:
            break

        if sample_size == prev_sample_size:
            if power > desired_power:
                sample_size -= 1
            else:
                sample_size += 1

        prev_sample_size = sample_size
        rich.print(f'sample size: {sample_size}\tpower: {power}\tstep: {step}')

        if step == max_step:
            rich.print('[bold red]Warning: max_step reached. '
                       'Sample size may not be optimized[/]')
            break

    if sample_size > max_sample_size:
        sample_size = max_sample_size
        power = sum((stat_test(
            *[i.sample(sample_size)
              for i in simulated_groups]).pvalue < alpha
            for _ in range(repetitions))) / repetitions

    return sample_size, power, effect_sizes


def get_stat_test_results(
    *data: list,
    alpha: float = .05,
    stat_test: Callable = stats.kruskal,
    effect_size_estimator: Callable = calc_dr,
    size_step: float = 10,
    n_tests: int = 1000,
        **kwargs) -> Tuple[int, float, List[float], List[float]]:
    '''
    **kwargs will go to get_optimal_sample_size function

    with small sample sizes statistic tests raise
    ValueError: All numbers are identical

    this function adjusts min_sample_size until it is sufficient
    and runs get_optimal_sample_size function followed by
    multiple stat_test calculations with obtained sample size

    returns tuple of sample size, power, effect sizes between groups
    and obtained statistic results
    '''
    success = False
    sample_size_min = 20
    while not success:
        try:
            optimal_size, power, effect_sizes = get_optimal_sample_size(
                *data, stat_test=stat_test, alpha=alpha,
                effect_size_estimator=effect_size_estimator,
                sample_size_min=sample_size_min, **kwargs)

            stat_res = [stat_test(*[i.sample(optimal_size)
                                    for i in data]) for _ in range(n_tests)]

            rich.print(f'optimal size: {optimal_size}\tpower: {power}\n')
            success = True
        except ValueError as e:
            if str(e).startswith('All numbers are identical'):
                rich.print('[red]no unique values during test![/]\t'
                           f'minimal sample size: {sample_size_min}')
                sample_size_min += size_step
            else:
                raise ValueError(e) from e

    return optimal_size, power, effect_sizes, stat_res


def perform_stat_with_subsampling(*data: list,
                                  alpha: float = .05,
                                  stat_test: Callable = stats.kruskal,
                                  need_posthoc: bool = True,
                                  posthoc_test: Callable =
                                  scikit_posthocs.posthoc_dunn,
                                  p_adjust: str = 'holm',
                                  n_tests: int = 1000,
                                  n_posthoc_tests: int = 100,
                                  effect_size_estimator: Callable = calc_dr,
                                  **kwargs
                                  ) -> namedtuple:
    '''
    get optimal sample size using subsampling
    then perform stat_test n_tests times.

    if need_posthoc:
    if stat_test is statistically significant,
    performs posthoc_test n_posthoc_tests times
    returns df with means and stds of p-values,sample_size, power, effect_sizes
    or stat_test wasn't significant means and stds of p-values will be the same

    if not need_posthoc:
    returns df with mean and std of p-values, sample_size, power, effect_size

    **kwargs will go to get_optimal_sample_size function

    returns namedtuple:
        'res' - pd.Dataframe with results,
        'stat_distr - statistic distribution
        'pvalue_distr' - p-value distribution of stat_test
    '''

    optimal_size, power, effect_sizes, stat_res = get_stat_test_results(
        *data,
        alpha=alpha,
        stat_test=stat_test,
        effect_size_estimator=effect_size_estimator,
        size_step=10,
        n_tests=n_tests,
        **kwargs)

    stat_vals = np.array([i.statistic for i in stat_res])
    pvals = np.array([i.pvalue for i in stat_res])
    mean_stat, std_stat = np.mean(stat_vals), np.std(stat_vals)
    mean_p, std_p = np.mean(pvals), np.std(pvals)

    Result = namedtuple('Result', ['res', 'stat_distr', 'pvalue_distr'])

    if need_posthoc:
        if mean_p < alpha:
            posthoc_pvals = [posthoc_test(
                [i.sample(optimal_size) for i in data],
                p_adjust=p_adjust)
                for _ in range(n_posthoc_tests)]
            pval_means = pd.concat(posthoc_pvals).groupby(level=0).mean()
            pval_stds = pd.concat(posthoc_pvals).groupby(level=0).std()
        else:
            # genersate similar tables to posthoc results but with
            # p-values from initial test
            pval_means = pd.DataFrame({
                i + 1: [mean_p] * i + [1] +
                [mean_p] * (len(data) - 1 - i)
                for i in range(len(data))})
            pval_means.index += 1
            pval_stds = pd.DataFrame({
                i + 1: [std_p] * i + [1] +
                [std_p] * (len(data) - 1 - i)
                for i in range(len(data))})
            pval_stds.index += 1

        df_dict = {'stat mean': [],
                   'stat std': [],
                   'p-value mean': [],
                   'p-value std': [],
                   'sample size': [],
                   'power': [],
                   'effect size': effect_sizes}
        index = []
        columns_checked = 0
        for i in pval_means.index:
            columns_checked += 1
            for j in pval_means.columns[columns_checked:]:
                df_dict['stat mean'].append(mean_stat)
                df_dict['stat std'].append(std_stat)
                df_dict['p-value mean'].append(pval_means.loc[i, j])
                df_dict['p-value std'].append(pval_stds.loc[i, j])
                df_dict['sample size'].append(optimal_size)
                df_dict['power'].append(power)
                if i > 1:
                    df_dict['effect size'].append(
                        effect_size_estimator(data[i - 1], data[j - 1]))
                index.append(f'{i} vs {j}')

        return Result(pd.DataFrame(df_dict, index), stat_vals, pvals)

    # return single row df if not need_posthoc
    return Result(pd.DataFrame({'stat mean': [mean_stat],
                                'stat std': [std_stat],
                                'p-value mean': [mean_p],
                                'p-value std': [std_p],
                                'sample size': [optimal_size],
                                'power': [power],
                                'effect size': [effect_sizes[0]]}),
                  stat_vals, pvals)


def bootstrap(*data: pd.Series,
              stat_test: Callable,
              n_resamples: int = 1000,
              confidence_level: float = .95,
              **kwargs) -> namedtuple:
    '''
    compute a two-sided bootstrap confidence interval of a statistic
    returns confidence_interval;
    use attributes low, high

    kwargs may be used in stat_test
    '''
    ConfidenceInterval = namedtuple('ConfidenceInterval',
                                    ['low', 'high'])
    if len(data) == 1:
        data = data[0]
        bootstrap_stats = [stat_test(data.sample(len(data), replace=True))
                           for _ in range(n_resamples)]

    elif len(data) == 0:
        raise ValueError('got empty array!')

    else:
        bootstrap_stats = np.zeros(n_resamples)
        for i in range(n_resamples):
            group_boots = [group.sample(len(group), replace=True)
                           for group in data]
            bootstrap_stats[i], _ = stat_test(*group_boots, **kwargs)

    percentile_low = (1 - confidence_level) / 2 * 100
    percentile_high = 100 - percentile_low

    return ConfidenceInterval(*np.percentile(
        bootstrap_stats, [percentile_low, percentile_high]))


def mw_critical_value(*data, alpha: float = .05) -> float:
    '''
    calculate critical value of mann whitney statistic
    '''
    n1 = len(data[0])
    n2 = len(data[1])
    # u_critical = min(statistic, n1 * n2 - statistic)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    return z_alpha * np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12) + (n2 - n1 / 2)


def kruskal_critical_value(*data, alpha: float = .05) -> float:
    '''
    calculate critical value of kruskal wallis statistic
    '''
    dof = len(data) - 1  # degrees of freedom
    return stats.chi2.ppf(1 - alpha, dof)


def ks_critical_value(*data, alpha: float = .05) -> float:
    '''
    calculate critical value of kolmogorov-smirnov statistic
    '''
    alpha = 0.05
    n1 = len(data[0])
    n2 = len(data[1])

    return stats.ksone.ppf(1 - alpha / 2, n1 + n2)
