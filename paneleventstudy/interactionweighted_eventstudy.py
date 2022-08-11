import pandas as pd
import numpy as np
from patsy import dmatrices
import statsmodels.formula.api as smf
from linearmodels import PanelOLS

from .dataprep import (
balancepanel,
checkcollinear,
checkfullrank,
)

def interactionweighted_eventstudy(
        data,
        outcome,
        event,
        group,
        cohort,
        reltime,
        calendartime,
        covariates,
        vcov_type='robust',
        check_balance=True
):
    print('\nEstimates an interaction-weighted event study regression as in Sun and Abraham (2021)')
    print('Ensure that reltime is int, with -1 = one period before treatment onset')
    print('Returns a dataframe containing the lead-lag coefficients and CIs')
    print('This version: vanilla off-the-shelf HAC-robust CIs')
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
    # Backing out cohort shares by reltime
    has_nevertreated = -1 in list(d[cohort])  # check if data set has a never-treated cohort
    if has_nevertreated:
        dnc = d[~(d[cohort] == -1)]  # exclude never-treated cohort
    if not has_nevertreated:
        dnc = d[~(d[cohort] == d[cohort].max())]  # exclude last-treated cohort
    # cohort shares (regression version)
    list_cohort = list(dnc[cohort].unique())
    cshares = pd.DataFrame(columns=[reltime, cohort, 'shares'])
    for c in list_cohort:
        dnc['cohort_ind'] = 0
        dnc.loc[dnc[cohort] == c, 'cohort_ind'] = 1  # dummy for if group belongs to cohort in question
        cs_eqn = 'cohort_ind ~ ' + 'C(' + reltime + ')' + ' - 1'  # no constant
        cs_mod = smf.ols(cs_eqn, data=dnc)
        cs_res = cs_mod.fit(vcov_type='HC3')  # robust standard errors
        cs_beta = pd.DataFrame(cs_res.params, columns=['shares'])
        # cs_resid = pd.DataFrame(cs_res.resid)  # not used in point estimates of IW coefficients
        cs_est = cs_beta.reset_index()
        cs_est[cohort] = c
        cs_est[reltime] = cs_est['index'].str.extract(
            '[C]\(' + reltime + '\)\[(.*?)\]',  # C(reltime)[?]
        )
        del cs_est['index']
        cshares = pd.concat([cshares, cs_est], axis=0).reset_index(drop=True)
    cshares[reltime] = cshares[reltime].astype('int')
    cshares[cohort] = cshares[cohort].astype('int')
    del dnc['cohort_ind']
    # Calculating asymptotic variance-covariance for the cohort shares (not available yet)
    # Estimate model
    d['Entity'] = d[group].copy()  # so that original entity column does not disappear
    d['Time'] = d[calendartime].copy()  # so that original calendar time column does not disappear
    d = d.set_index(['Entity', 'Time'])  # entity (outer) - time (inner) multiindex for linearmodels.PanelOLS
    min_reltime = d[reltime].min()  # backs out the smallest value in the reltime column
    d.loc[d[reltime] == -1, reltime] = min_reltime - 100  # low-tech bypass to set -1 as the reference; C(x, Treatment('-1')) is not working?!
    # Prepare design matrices for estimating CATTs
    if len(covariates) == 0:
        eqn = outcome + " ~ " + "C(" + reltime + ")" + ":" + "C(" + cohort + ")"  # Need to include intercept
    elif len(covariates) > 0:
        eqn = outcome + " ~ " + "C(" + reltime + ")" + ":" + "C(" + cohort + ")" + " +" + "+".join(covariates)
    print('Estimating equation: ' + eqn)
    # Copy of CATT column labels to be kept later
    yint, Xint = dmatrices(
        outcome + " ~ " + "C(" + reltime + ")" + ":" + "C(" + cohort + ")",  # no intercept
        d,
        return_type='dataframe'
    )
    del yint  # redundant
    catt_keep = list(Xint.columns)  # list to be used to keep only CATTs (cohort-specific lead-lag coefficients)
    catt_keep.remove('Intercept')
    del Xint
    # Main design matrix
    y, X = dmatrices(
        eqn,
        d,
        return_type='dataframe'
    )
    # Check for full rank in X
    list_X = list(X.columns)
    drop_X = checkfullrank(data=X, rhs=list_X)  # exclude intercept from the check
    for x in drop_X:
        del X[x]
    print('The following columns were dropped due to linear dependence:\n' + ', '.join(drop_X))
    # Check for collinearity and duplicates in X
    list_X = list(X.columns)
    drop_X = checkcollinear(data=X, rhs=list_X)  # exclude intercept from the check
    for x in drop_X:
        del X[x]
    print('The following columns were dropped due to collinearity or duplication:\n' + ', '.join(drop_X))
    # Estimate CATTs
    catt_mod = PanelOLS(
        dependent=y,
        exog=X,
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True
    )  # Also prints out absorbed columns (collinear with time or entity effects)
    catt_res = catt_mod.fit(cov_type=vcov_type)
    catt_beta = pd.DataFrame(catt_res.params)  # all estimated coefficients
    catt_ci = pd.DataFrame(catt_res.conf_int())  # CIs of all estimated coefficients
    catt_est = catt_beta.merge(catt_ci, how='outer', left_index=True, right_index=True)
    # catt_varcov = catt_res.cov  # variance-covariance of estimators (not used for now)
    # Calculate interaction-weighted ATTs
    iw_est = catt_est.reset_index()
    iw_est = iw_est[iw_est['index'].isin(catt_keep)]
    iw_est[reltime] = iw_est['index'].str.extract(
        '[C]\(' + reltime + '\)\[T\.(.*?)\]',  # C(reltime)[T.?]:C(cohort)[?]
    )
    iw_est[cohort] = iw_est['index'].str.extract(
        '[C]\(' + cohort + '\)\[(.*?)\]',  # C(reltime)[T.?]:C(cohort)[?]
    )
    iw_est[[reltime, cohort]] = iw_est[[reltime, cohort]].astype('int')
    iw_est = iw_est[[reltime, cohort, 'parameter', 'lower', 'upper']]  # rearrange columns + drop old index column
    iw_est = iw_est.merge(cshares, how='outer', on=[reltime, cohort])  # merge on reltime-cohort
    iw_est = iw_est[~iw_est['parameter'].isna()].reset_index(drop=True)  # drop rows for omitted periods / cohorts (typically reltime = -1)
    iw_est['parameter'] = iw_est['parameter'] * iw_est['shares']  # CATT * weight
    iw_est['lower'] = iw_est['lower'] * iw_est['shares']  # CATT * weight
    iw_est['upper'] = iw_est['upper'] * iw_est['shares']  # CATT * weight
    iw_est = iw_est.groupby(reltime)[['parameter', 'lower', 'upper', 'shares']].agg('sum')  # sum(CATT * weight) across cohorts
    # Correcting relic of dropped cohort * reltime terms
    for i in ['parameter', 'lower', 'upper']:
        iw_est.loc[iw_est['shares'] < 1, i] = iw_est[i] / iw_est['shares']
    del iw_est['shares']
    return iw_est