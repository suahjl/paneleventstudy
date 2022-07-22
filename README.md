# paneleventstudy
This Python package implements the panel (and single entity) event study models, covering the naive two-way fixed effects implementation, and the interaction-weighted implementation from Sun and Abraham (2021) (derived from https://github.com/lsun20/EventStudyInteract).

The package includes threesets of functions:
1. ```Data cleaning```: Functions to prepare data frames for the analytical set of functions, e.g., ensuring that they are in the right format, and have the right columns (with the right content)
2. ```Analytical```: Direct implementation of the event study models
3. ```Utilities```: Tools to assist the user in setting up input-output flows
 
# Installation
1. ```pip install paneleventstudy```


# Implementation (Data Cleaning)
## Counting and dropping missing observations
### Documentation
```python
paneleventstudy.dropmissing(data, event)
```
#### Parameters
#### Output
### Example

## Checking if input dataframe is a balanced panel
### Documentation
```python
paneleventstudy.balancepanel(data, group, event, calendartime, check_minmax=True)
```
#### Parameters
#### Output
### Example

## Generate column indicating control groups (never-treated / last-treated)
### Documentation
```python
paneleventstudy.identifycontrols(data, group, event)
```
#### Parameters
#### Output
### Example

## Generate relative time column (treatment onset = 0)
### Documentation
```python
paneleventstudy.genreltime(data, group, event, calendartime, reltime='reltime', check_balance=True)
```
#### Parameters
#### Output
### Example

## Generate column indicating treatment cohorts
### Documentation
```python
paneleventstudy.gencohort(data, group, event, calendartime, cohort='cohort', check_balance=True)
```
#### Parameters
#### Output
### Example

## Generate calendar time with integers
### Documentation
```python
paneleventstudy.gencalendartime_numerics(data, group, event, calendartime, calendartime_numerics='ct')
```
#### Parameters
#### Output
### Example

## Identify collinear or invariant columns
### Documentation
```python
paneleventstudy.checkcollinear(data, rhs)
```
#### Parameters
#### Output
### Example

## Identify linearly dependent columns
### Documentation
```python
paneleventstudy.checkfullrank(data, rhs, intercept='Intercept')
```
#### Parameters
#### Output
### Example

# Implementation (Analytical)
## Naive TWFE Panel Event Study
Estimates dynamic treatment effects ($\beta_{l}$ coefficients on leads and lags of treatment dummies) using a standard two-way fixed effects model, where $l=0$ refers to when the treatment was applied to entity $i$.

$$Y_{i,t} = \alpha_i + \alpha_t + \sum_{l=-K}^{-2} \beta_{l} D_{i, t}^{l} + \sum_{l=0}^{M} \beta_{l} D_{i, t}^{l} + \epsilon_{i, t}$$

### Documentation
```python
paneleventstudy.naivetwfe_eventstudy(data, outcome, event, group, reltime, calendartime, covariates, vcov_type='robust', check_balance=True)
```
#### Parameters
#### Output
### Example


## Interaction-Weighted Panel Event Study 
### Documentation
```python
paneleventstudy.interactionweighted_eventstudy(data, outcome, event, group, cohort, reltime, calendartime, covariates, vcov_type='robust', check_balance=True)
```
#### Parameters
#### Output
### Example


## Single Entity Event Study 
### Documentation
```python
paneleventstudy.timeseries_eventstudy(data, outcome, reltime, covariates, vcov_type='HC3')
```
#### Parameters
#### Output
### Example

# Implementation (Utilities)
## Plotting Event Study Lead and Lag Coefficients
### Documentation
```python
paneleventstudy.eventstudyplot(input, big_title='Event Study Plot (With 95% CIs)', path_output='', name_output='eventstudyplot')
```
#### Parameters
#### Output
### Examples


# Requirements
## Python Packages
- pandas>=1.4.3
- numpy>=1.23.0
- linearmodels>=4.27
- plotly>=5.9.0
- statsmodels>=0.13.2
- sympy>=1.10.1
