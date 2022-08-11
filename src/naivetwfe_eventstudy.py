import pandas as pd
import numpy as np
from linearmodels import PanelOLS

from dataprep import (
balancepanel,
)

def naivetwfe_eventstudy(
        data,
        outcome,
        event,
        group,
        reltime,
        calendartime,
        covariates,
        vcov_type='robust',
        check_balance=True
):
    print('\nEstimates a naive event study regression using dynamic TWFE')
    print('Ensure that reltime is int, with -1 = one period before treatment onset')
    print('Returns a dataframe containing the lead-lag coefficients and CIs')
    d = data.copy()
    # Check if reltime is integer
    if d[reltime].dtypes == int:
        pass
    elif ~(d[reltime].dtypes == int):
        raise NotImplementedError('reltime must be integer')
    # Run interim check if panel is balanced
    if check_balance:
        check_balancepanel = balancepanel(data=data, group=group, event=event, calendartime=calendartime, check_minmax=False)
        if check_balancepanel:
            print('Quick check indicates panel is balanced')
        elif not check_balancepanel:
            print('Panel is NOT balanced')
            raise NotImplementedError  # Since this is general to any format, panel MUST be balanced ex ante
    # Estimate model
    d['Entity'] = d[group].copy()  # so that original entity column does not disappear
    d['Time'] = d[calendartime].copy()  # so that original calendar time column does not disappear
    d = d.set_index(['Entity', 'Time'])  # entity (outer) - time (inner) multiindex for linearmodels.PanelOLS
    min_reltime = d[reltime].min()  # backs out the smallest value in the reltime column
    d.loc[d[reltime] == -1, reltime] = min_reltime - 100  # low-tech bypass to set -1 as the reference; C(x, Treatment('-1')) is not working?!
    if len(covariates) == 0:
        eqn = outcome + " ~ 1 + " + "C(" + reltime + ")" + " + EntityEffects + TimeEffects"  # Need to include intercept
    elif len(covariates) > 0:
        eqn = outcome + " ~ 1 + " + "C(" + reltime + ")" + " +" + "+".join(covariates) + " + EntityEffects + TimeEffects" # R-style equations
    print('Estimating equation: ' + eqn)
    mod = PanelOLS.from_formula(eqn, data=d, drop_absorbed=True)
    res = mod.fit(cov_type=vcov_type)
    beta = pd.DataFrame(res.params)  # all estimated coefficients
    ci = pd.DataFrame(res.conf_int())  # CIs of all estimated coefficients
    est = beta.merge(ci, how='outer', left_index=True, right_index=True)
    # Clean up frame containing estimated parameters
    key_reltime = 'C(' + reltime + ')[T.'
    est = est.reset_index(drop=False)
    est = est[est['index'].str.contains(key_reltime, regex=False)]  # not Regex
    est['index'] = est['index'].str.replace(key_reltime, '', regex=False)  # not Regex
    est['index'] = est['index'].str.replace(']', '', regex=False)  # not Regex
    est = est.set_index('index')
    return est