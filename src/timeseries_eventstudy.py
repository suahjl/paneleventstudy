import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def timeseries_eventstudy(
        data,
        outcome,
        reltime,
        covariates,
        vcov_type='HC3',
):
    print('\nEstimates an event study regression for a single entity setting using OLS')
    print('Ensure that reltime is int, with -1 = one period before treatment onset')
    print('Returns a dataframe containing the lead-lag coefficients and CIs')
    d = data.copy()
    # Check if reltime is integer
    if d[reltime].dtypes == int:
        pass
    elif ~(d[reltime].dtypes == int):
        raise NotImplementedError('reltime must be integer')
    min_reltime = d[reltime].min()  # backs out the smallest value in the reltime column
    d.loc[d[reltime] == -1, reltime] = min_reltime - 100  # low-tech bypass to set -1 as the reference; C(x, Treatment('-1')) is not working?!
    if len(covariates) == 0:
        eqn = outcome + " ~ " + "C(" + reltime + ")"  # Don't have to explicitly declare intercept
    elif len(covariates) > 0:
        eqn = outcome + " ~ " + "C(" + reltime + ")" + " +" + "+".join(covariates)  # R-style equations
    print('Estimating equation: ' + eqn)
    # Estimating single entity event study model
    mod = smf.ols(eqn, data=d)
    res = mod.fit(vcov_type=vcov_type)
    beta = pd.DataFrame(res.params, columns=['parameter'])
    ci = pd.DataFrame(res.conf_int(), columns=['lower', 'upper'])
    est = beta.merge(ci, how='left', left_index=True, right_index=True)
    # Clean up frame containing estimated parameters
    key_reltime = 'C(' + reltime + ')[T.'
    est = est.reset_index(drop=False)
    est = est[est['index'].str.contains(key_reltime, regex=False)]  # not Regex
    est['index'] = est['index'].str.replace(key_reltime, '', regex=False)  # not Regex
    est['index'] = est['index'].str.replace(']', '', regex=False)  # not Regex
    est = est.set_index('index')
    return est