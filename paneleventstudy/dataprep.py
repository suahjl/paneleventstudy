# Scripts to clean and prepare data for the analytical functions

import pandas as pd
import numpy as np
import sympy

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
    print('Transforming input matrix into reduced row echelon form')
    rf, ind = sympy.Matrix(d.values).rref()  # reduced row echelon form, locations of linearly independent columns
    ind = list(ind)  # convert tuple into list
    d = d.iloc[:, ind]  # keep only linearly independent columns
    col_d_ind = list(d.columns)  # list of linearly independent columns
    col_d_dep = list(set(rhs) ^ set(col_d_ind)) # reverse engineer a list of columns in RHS (original input) that should be dropped
    return col_d_dep