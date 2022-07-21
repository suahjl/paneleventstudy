### Replicates interaction-weighted event study estimator from Sun and Abraham (2020)
### https://github.com/lsun20/EventStudyInteract/blob/main/eventstudyinteract.ado
### Test naive event study
### Uses daily transactions data

import pandas as pd
import numpy as np
from datetime import date
from patsy import dmatrices
import sympy
import statsmodels.formula.api as smf
from linearmodels import PanelOLS
import plotly.graph_objects as go
from tqdm import tqdm
import telegram_send

### Preliminaries
## Data frame
df = pd.read_excel('transactions_location.xlsx', sheet_name='sa')  # seasonally adjusted state-level card transactions
del df['online']  # delete online spending
df = df.add_prefix('spending')  # to ease wide_to_long later
df = df.rename(columns={'spendingUnnamed: 0': 'date'})  # blank excel column label
df['date'] = df['date'].dt.date  # from datetime to date format
df = pd.wide_to_long(df=df, stubnames='spending', i='date', j='state', suffix=r'\w+')
df = df.reset_index()
## Normalise state-level data points
list_state = list(df['state'].unique())
ref_date = date(2020,1,1)
for i in tqdm(list_state):
    ref_spend = df.loc[((df['date'] == ref_date) & (df['state'] == i)), 'spending'].reset_index(drop=True)[0]  # reference spending for state
    df.loc[df['state'] == i, 'spending'] = 100 * (df.loc[df['state'] == i, 'spending'] / ref_spend)  # 100-indexed to ref_date
## Flood timings
dict_state_flooddate = {'Perlis': '',
                        'Kedah': '',
                        'Penang': '',
                        'Kelantan': date(2021, 12, 16),
                        'Terengganu': date(2021, 12, 16),
                        'Pahang': date(2021, 12, 19),
                        'Perak': date(2021, 12, 18),
                        'Selangor': date(2021, 12, 17),
                        'KualaLumpur': date(2021, 12, 17),
                        'Putrajaya': date(2021, 12, 17),
                        'NegeriSembilan': date(2021, 12, 18),
                        'Melaka': '',
                        'Johor': '',
                        'Labuan': '',
                        'Sabah': '',
                        'Sarawak': ''}
for i,j in tqdm(dict_state_flooddate.items()):
    df.loc[((df['state'] == i) & (df['date'] == j)), 'flood'] = 1
df['flood'] = df.groupby('state')['flood'].fillna(method='ffill')
df['flood'] = df['flood'].fillna(0)
df['flood'] = df['flood'].astype('int')
## Study period
df = df[((df['date'] >= date(2021, 12, 5)) &
         (df['date'] <= date(2021, 12, 24)))].reset_index(drop=True)

### Package drafts
def dropmissing(data, event):
    print('\nChecking for and dropping missing observations')
    d = data.copy()
    print('Number of observations in input dataframe: ' + str(d.iloc[:, 0].count()))  # counts only the first column
    d = d[~(d[event].isna())]
    print('Number of observations in output dataframe: ' + str(d.iloc[:, 0].count()))  # counts only the first column
    return d

def balancepanel(data, group, event, calendartime, check_minmax=True):
    print('\nChecks if input data frame is a BALANCED PANEL')
    d = data.copy()
    print('Checking if every group has the same number of observations')
    check_N = d.groupby(group)[event].count()
    check_N = check_N.nunique()
    if check_N == 1:
        print('Every group has the same number of observations')
    elif check_N > 1:
        print('Some groups have different numbers of observations!')
    if check_minmax: # calendartime is ignored if check_minmax = False
        print('Checking min and max of calendartime by groups (ONLY for datetime/int)')
        print('Checking if the smallest calendartime is the same for all groups (ONLY for datetime/int)')
        check_min = d.groupby(group)[calendartime].min()
        check_min = check_min.nunique()
        if check_min == 1:
            print('Every group has the same minimum calendartime value')
        elif check_min > 1:
            print('Some groups have different minimum calendartime values!')
        print('Checking if the largest calendartime is the same for all groups (ONLY for datetime/int)')
        check_max = d.groupby(group)[calendartime].max()
        check_max = check_max.nunique()
        if check_max == 1:
            print('Every group has the same maximum calendartime value')
        elif check_max > 1:
            print('Some groups have different maximum calendartime values!')
        if (check_N == 1) & (check_min == 1) & (check_max == 1):
            check_balancepanel = True
        elif ~((check_N == 1) & (check_min == 1) & (check_max == 1)):
            check_balancepanel = False
    elif not check_minmax:
        print('NOT checking min and max of calendartime by groups')
        if check_N == 1 :
            check_balancepanel = True
        elif ~(check_N == 1):
            check_balancepanel = False
    return check_balancepanel  # returns boolean indicating if the panel is balanced

def identifycontrols(data, group, event):
    print('\nIdentifying control groups: never-treated, or last-treated')
    print('Generates new indicator column: control_group')
    d = data.copy()
    list_group = list(d[group].unique())  # generate iterable of all levels in the group column
    ## Finding never-treated groups
    print('Finding never-treated groups')
    for g in list_group:  # finding if there are any groups with event = 0 for all calendartime
        d.loc[d[group] == g, '_eventmax'] = d.loc[d[group] == g, event].max()
        d.loc[((d[group] == g) & (d['_eventmax'] == 0)), 'control_group'] = 1
        d.loc[((d[group] == g) & (d['_eventmax'] == 1)), 'control_group'] = 0
    del d['_eventmax']
    ## Finding last-treated group
    check_control = d['control_group'].max()
    if check_control == 1:
        print('There exists a never-treated group, skipping the search for last-treated groups')
    elif check_control == 0:
        print('No never-treated groups exist, proceeding to find last-treated groups')
        for g in list_group:  # Quick hack: Last-treated groups have the least number of event = 1
            d.loc[d[group] == g, '_event_flagavg'] = d.loc[d[group] == g, event].mean()  # average number of event flags by group
        min_event_flagavg = d['_event_flagavg'].min()  # find the global minimum avg event flag count
        d.loc[d['_event_flagavg'] == min_event_flagavg, 'control_group'] = 1  # check which group made the minimum avg event flag
        for g in list_group:  # make the control group indicator time-invariant by group
            d.loc[d[group] == g, 'control_group'] = d.loc[d[group] == g, 'control_group'].max()
        del d['_event_flagavg']
    d['control_group'] = d['control_group'].astype('int') # set as integer
    return d

def genreltime(data, group, event, calendartime, reltime='reltime', check_balance=True):
    print('\nGenerating relative time columns from event column; time0 = when treatment is applied')
    print("To generalise, calendartime's format will be ignored")
    print("PLEASE ensure that calendartime is ascending without gaps, i.e., T-k, T-k+1, ..., T, ..., T+k")
    print("Input dataframe must be a BALANCED PANEL")
    print('Relative time will be stored in ' + reltime)
    d = data.copy()
    list_group = list(d[group].unique())
    if check_balance:  # run interim check if panel is balanced
        check_balancepanel = balancepanel(data=data, group=group, event=event, calendartime=calendartime, check_minmax=False)
        if check_balancepanel:
            print('Quick check indicates panel is balanced')
        elif not check_balancepanel:
            print('Panel is NOT balanced')
            raise NotImplementedError  ## Since this is generalised to any calendartime format, panel MUST be balanced ex ante
    for g in list_group:  # create new temporary column of calendar time going from 0 to end of time
        d.loc[d[group] == g, '_ct'] = np.arange(len(d[d[group] == g]))
    for g in list_group:  # create new column with calendar - onset as integers
        try:
            d.loc[d[group] == g, reltime] = \
                d.loc[d[group] == g, '_ct'] - d.loc[((d[group] == g) & (d[event] == 1)), '_ct'].reset_index(drop=True)[0]
        except:
            print('Group ' + g + ' has no events')
    print('All groups without events have ' + reltime + ' filled with 0')
    d.loc[d[reltime].isna(), reltime] = 0  # vectorised for speed
    d[reltime] = d[reltime].astype('int') # set as integer
    del d['_ct']  # delete temporary column
    return d

def gencohort(data, group, event, calendartime, cohort='cohort', check_balance=True):
    print('\nGenerating cohort indicators')
    print("To generalise, calendartime's format will be ignored")
    print("PLEASE ensure that calendartime is ascending without gaps, i.e., T-k, T-k+1, ..., T, ..., T+k")
    print("Input dataframe must be a BALANCED PANEL")
    print('Cohort indicators will be stored in ' + cohort)
    d = data.copy()
    list_group = list(d[group].unique())
    if check_balance:  # run interim check if panel is balanced
        check_balancepanel = balancepanel(data=data, group=group, event=event, calendartime=calendartime, check_minmax=False)
        if check_balancepanel:
            print('Quick check indicates panel is balanced')
        elif not check_balancepanel:
            print('Panel is NOT balanced')
            raise NotImplementedError  ## Since this is generalised to any calendartime format, panel MUST be balanced ex ante
    for g in list_group:  # create new temporary column of calendar time going from 0 to end of time
        d.loc[d[group] == g, '_ct'] = np.arange(len(d[d[group] == g]))
    for g in list_group:
        try:
            d.loc[d[group] == g, cohort] = \
                d.loc[((d[group] == g) & (d[event] == 1)), '_ct'].reset_index(drop=True)[0]  # onset time by group
        except:
            print('Group ' + g + ' has no events')
    print('All groups without events have ' + cohort + ' filled with -1')
    d.loc[d[cohort].isna(), cohort] = -1  # vectorised for speed
    d[cohort] = d[cohort].astype('int') # set as integer
    del d['_ct']  # delete temporary column
    return d

def gencalendartime_numerics(data, group, event, calendartime, calendartime_numerics='ct'):
    print('\nGenerating numerics calendar time from 0 to T (end of time)')
    print('Intended to make calling PanelOLS dependencies later easier by bypassing the datetime requirement')
    print("PLEASE ensure that calendartime is ascending without gaps, i.e., T-k, T-k+1, ..., T, ..., T+k")
    print("Input dataframe must be a BALANCED PANEL")
    d = data.copy()
    list_group = list(d[group].unique())
    check_balancepanel = balancepanel(data=data, group=group, event=event, calendartime=calendartime, check_minmax=False)
    if check_balancepanel:
        print('Quick check indicates panel is balanced')
    elif not check_balancepanel:
        print('Panel is NOT balanced')
        raise NotImplementedError  # Since this is generalised to any calendartime format, panel MUST be balanced ex ante
    for g in list_group:  # create new column of calendar time going from 0 to end of time
        d.loc[d[group] == g, calendartime_numerics] = np.arange(len(d[d[group] == g]))
    d[calendartime_numerics] = d[calendartime_numerics].astype('int')  # set as integer
    return d

def checkcollinear(data, rhs):
    print('\nChecks if RHS variables contained within the dataframe are collinear or invariant')
    print('Will trim collinear (-1 / +1 in correlation matrix) or invariant (nan in correlation matrix) columns')
    print('Later columns get precedence')
    print('Returns list of columns that should be dropped in data[rhs]')
    d = data.copy()
    abscor = np.abs(d[rhs].corr())
    # Backing out which column pairs have absolute correlation = 1
    list_violate = []
    for x in rhs:
        list_Z = rhs.copy()  # in-loop copy of fill list of RHS variables
        list_Z.remove(x)  # remove the variable in consideration
        for z in list_Z:
            rho = abscor.loc[x, z]
            if (rho == 1) | (rho == np.nan):  # either collinear or invariant
                list_violate = list_violate + [x]  # drop x if precedence goes to later columns; drop z if otherwise
                list_Z.remove(z)  # since dropped, remove from comparator list permanently
            elif (rho < 1) & ~(rho == np.nan):  # neither collinear nor invariant
                pass
    # Backing out which columns are duplicates of another columns
    bool_dupe = d[rhs].T.duplicated()
    if len(bool_dupe) > 0:
        bool_dupe = bool_dupe[bool_dupe]
        list_violate_dupe = list(bool_dupe.index)
    elif len(bool_dupe) == 0:
        list_violate_dupe = []
    # Ensure that final list has unique elements
    if (len(list_violate_dupe) > 0) & (len(list_violate) > 0):
        for i in list_violate_dupe:
            try:
                list_violate.remove(i)
            except:
                pass
        list_violate = list_violate + list_violate_dupe
    elif (len(list_violate_dupe) == 0) & (len(list_violate) > 0):
        list_violate = list_violate
    elif (len(list_violate_dupe) > 0) & (len(list_violate) == 0):
        list_violate = list_violate_dupe
    return list_violate

def checkfullrank(data, rhs, intercept='Intercept'):
    print('\nChecks if RHS columns in data are linearly independent')
    print('Will trim columns that are linearly dependent')
    print('Later columns get precedence')
    print('Returns a list of columns that are linearly dependent, and should be dropped')
    rhs_reverse = rhs.copy()  # deep copy, then reverse later
    rhs_reverse.reverse()  # reverses on self (if intercept is given, it will now be at the end)
    # Check if intercept is given
    if intercept is None:
        pass
    elif intercept is not None:
        print('Intercept label given; intercept column will be given precedence')
        rhs_reverse.remove(intercept)
        rhs_reverse = [intercept] + rhs_reverse  # move intercept back to the very front of the reversed list (prioritised)
    d = data[rhs_reverse].copy()  # deep copy (reversed columns to be checked for full rank)
    # Identify linearly dependent columns
    rf, ind = sympy.Matrix(d.values).rref()  # reduced row echelon form, locations of linearly independent columns
    ind = list(ind)  # convert tuple into list
    d = d.iloc[:, ind]  # keep only linearly independent columns
    col_d_ind = list(d.columns)  # list of linearly independent columns
    col_d_dep = list(set(rhs) ^ set(col_d_ind)) # reverse engineer a list of columns in RHS (original input) that should be dropped
    return col_d_dep

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
    mod = PanelOLS.from_formula(eqn, data=d)
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

def eventstudyplot(input, big_title='Event Study Plot (With 95% CIs)', path_output='', name_output='eventstudyplot'):
    print('\nTakes output from est_ functions to plot event study estimates and their CIs')
    print('Requires 3 columns: parameter, lower, and upper; indexed to relative time')
    print('Output = plotly graph objects figure')
    d = input.copy()
    fig = go.Figure()
    # Point estimate
    fig.add_trace(
        go.Scatter(
            x=d.index,
            y=d['parameter'],
            name='Coefficients on Lead / Lags of Treatment',
            mode='lines',
            line=dict(color='black', width=3)
        )
    )
    # Lower bound
    if len(d[~d['lower'].isna()]) > 0:
        fig.add_trace(
            go.Scatter(
                x=d.index,
                y=d['lower'],
                name='Lower Confidence Bound',
                mode='lines',
                line=dict(color='black', width=1, dash='dash')
            )
        )
    # Upper bound
    if len(d[~d['upper'].isna()]) > 0:
        fig.add_trace(
            go.Scatter(
                x=d.index,
                y=d['upper'],
                name='Upper Confidence Bound',
                mode='lines',
                line=dict(color='black', width=1, dash='dash')
            )
        )
    # Overall layout
    fig.update_layout(
        title=big_title,
        plot_bgcolor='white',
        font=dict(color='black')
    )
    # Save output
    fig.write_html(path_output + name_output + '.html')
    fig.write_image(path_output + name_output + '.png', height=768, width=1366)
    # fig.show()
    return fig

def telsendimg(conf='', path='', cap=''):
    with open(path, 'rb') as f:
        telegram_send.send(conf=conf,
                           images=[f],
                           captions=[cap])
def telsendfile(conf='', path='', cap=''):
    with open(path, 'rb') as f:
        telegram_send.send(conf=conf,
                           files=[f],
                           captions=[cap])

### Test package
df = dropmissing(data=df, event='flood')
check_balancepanel = balancepanel(data=df, group='state', event='flood', calendartime='date')
if not check_balancepanel:
    raise NotImplementedError
df = identifycontrols(data=df, group='state', event='flood')
df = genreltime(data=df, group='state', event='flood', calendartime='date', reltime='reltime', check_balance=True)
df = gencohort(data=df, group='state', event='flood', calendartime='date', cohort='cohort', check_balance=True)
df = gencalendartime_numerics(data=df, group='state', event='flood', calendartime='date', calendartime_numerics='ct')
est_twfe = naivetwfe_eventstudy(
    data=df,
    outcome='spending',
    event='flood',
    group='state',
    reltime='reltime',
    calendartime='ct',
    covariates=[],
    vcov_type='robust',
    check_balance=True
)
fig_twfe = eventstudyplot(
    input=est_twfe,
    big_title='Naive TWFE Event Study Plot (With 95% CIs)',
    name_output='eventstudyplot_twfe'
)
est_iw = interactionweighted_eventstudy(
    data=df,
    outcome='spending',
    event='flood',
    group='state',
    cohort='cohort',
    reltime='reltime',
    calendartime='ct',
    covariates=[],
    vcov_type='robust',
    check_balance=True
)
fig_iw = eventstudyplot(
    input=est_iw,
    big_title='Sun & Abraham (2021) Interaction-Weighted Event Study Plot (With 95% CIs)',
    name_output='eventstudyplot_iw'
)

option_ci = 0
fig = go.Figure()
# TWFE
fig.add_trace(
    go.Scatter(
        x=est_twfe.index,
        y=est_twfe['parameter'],
        name='TWFE Estimates',
        mode='lines',
        line=dict(color='red', width=2)
    )
)
if option_ci == 1:
    fig.add_trace(
            go.Scatter(
                x=est_twfe.index,
                y=est_twfe['lower'],
                name='TWFE CIs',
                mode='lines',
                line=dict(color='red', width=1, dash='dash')
            )
        )
    fig.add_trace(
            go.Scatter(
                x=est_twfe.index,
                y=est_twfe['upper'],
                name='TWFE CIs',
                mode='lines',
                line=dict(color='red', width=1, dash='dash')
            )
        )
# IW
fig.add_trace(
    go.Scatter(
        x=est_iw.index,
        y=est_iw['parameter'],
        name='Interaction-Weighted Estimates',
        mode='lines',
        line=dict(color='black', width=2)
    )
)
if option_ci == 1:
    fig.add_trace(
            go.Scatter(
                x=est_iw.index,
                y=est_iw['lower'],
                name='IW CIs',
                mode='lines',
                line=dict(color='black', width=1, dash='dash')
            )
        )
    fig.add_trace(
            go.Scatter(
                x=est_iw.index,
                y=est_iw['upper'],
                name='IW CIs',
                mode='lines',
                line=dict(color='black', width=1, dash='dash')
            )
        )
# Overall layout
fig.update_layout(
    title='TWFE v IW Event Study Estimates: Effect of Floods in Mid-Dec 2021 on Card Spending (State-Level); Event Onset: T=0',
    plot_bgcolor='white',
    font=dict(color='black')
)
# Save output
fig.write_html('eventstudyplot_compare' + '.html')
fig.write_image('eventstudyplot_compare' + '.png', height=768, width=1366)
# telsendimg(conf='EcMetrics_Config_GeneralFlow.conf',
#            path='eventstudyplot_compare' + '.png',
#            cap='Classic Two-Way Fixed Effects v. Interaction-Weighted Event Study\n\n' +
#                'Effect of Floods in Dec 2021 on Card Spending (State-Level); Event Onset: T=0')

# Check out single entity version
df_ts = df[df['state'] == 'Selangor']
df_ts = df_ts.reset_index(drop=True)
est_ts = timeseries_eventstudy(
    data=df_ts,
    outcome='spending',
    reltime='reltime',
    covariates=[],
    vcov_type='HC0'
)
fig_ts = eventstudyplot(
    input=est_ts,
    big_title='Single Entity Event Study Plot (With 95% CIs): Selangor Dec 2021 Flood',
    name_output='eventstudyplot_ts'
)

############## JUNKYARD

# cohort shares (tabulate version; not used)
# del cshares
# cshares = pd.DataFrame(columns=[reltime, cohort, 'shares'])
# cshares[[reltime, cohort]] = cshares[[reltime, cohort]].astype('int')
# list_reltime = list(dnc[reltime].unique())
# for r in list_reltime:
#     dr = dnc.loc[dnc[reltime] == r]
#     n_by_r = dr[event].count()
#     cs = dr.groupby(cohort)[event].agg('count') / n_by_r
#     cs = cs.reset_index().rename(columns={event: 'shares'})
#     cs[reltime] = r
#     cshares = pd.concat([cshares, cs], axis=0).reset_index(drop=True)
# return cshares