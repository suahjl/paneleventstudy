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
## Naive Two-Way Fixed Effects (TWFE) Panel Event Study
Estimates dynamic treatment effects using a standard TWFE model. 
Specifically, we are interested in estimating $\beta_{l}$, the coefficients on leads and lags of treatment dummies, where $l$ is relative time as in [Sun and Abraham (2021)](https://doi.org/10.1016/j.jeconom.2020.09.006), i.e., the time period relative to treatment onset. 
$l=0$ refers to when the treatment was applied to entity $i$.

$D_{i,t}^{l}$ are dummies switching on if entity $i$ is in calendar time $t$, and is $l$ periods relative to the treatment onset. That is also to say that $D_{i, t}^{l} \ \forall \ t, l$ never-treated entities will take values $0$.
A TWFE regression model includes entity fixed effects ($\alpha_i$), and time fixed effects ($\alpha_t$). $\mathbf{X_{i, t}}$ is an optional vector of time-varying (within-entity) controls.
$\epsilon_{i, t}$ are the errors.

$$Y_{i,t} = \alpha_i + \alpha_t + \sum_{l=-K}^{-2} \beta_{l} D_{i, t}^{l} + \sum_{l=0}^{M} \beta_{l} D_{i, t}^{l} + \mathbf{X_{i, t} \gamma} + \epsilon_{i, t}$$

### Documentation
```python
paneleventstudy.naivetwfe_eventstudy(data, outcome, event, group, reltime, calendartime, covariates, vcov_type='robust', check_balance=True)
```
#### Parameters
#### Output
### Example


## Interaction-Weighted Panel Event Study 
Estimates dynamic treatment effects using the interaction-weighted estimator described in [Sun and Abraham (2021)](https://doi.org/10.1016/j.jeconom.2020.09.006).
Again, for the following structural equation, we are interested in estimating $\beta_{l}$, the coefficients on leads and lags of treatment dummies, where $l$ is relative time as in [Sun and Abraham (2021)](https://doi.org/10.1016/j.jeconom.2020.09.006), i.e., the time period relative to treatment onset. 
$l=0$ refers to when the treatment was applied to entity $i$.

$$Y_{i,t} = \alpha_i + \alpha_t + \sum_{l=-K}^{-2} \beta_{l} D_{i, t}^{l} + \sum_{l=0}^{M} \beta_{l} D_{i, t}^{l} + \mathbf{X_{i, t} \gamma} + \eta_{i, t}$$

This implementation has xx broad steps.

1. Calculate the cohort shares by relative time, $\mathbb{E} (E_i = e | E_i \in g )$ where $g$ is the set of relative times included in the analysis. This package uses a no-constant linear regression model with an OLS estimator as per the Sun and Abraham (2021)'s original Stata package [here](https://github.com/lsun20/EventStudyInteract). Using a linear regression approach, instead of simple tabulation, allows for calculation of standard errors of the cohort share estimates.
	$$ 1\{E_i = e | E_i \in g \} = w_i  + e_i $$
	
2. Estimate the cohort-specific average treatment effects, ${CATT}_{e, l}$, by interacting the cohort dummy with the treatment / relative time dummy, $1{E_i = e} D_{i,t}^{l}$.
	$$Y_{i,t} = \alpha_i + \alpha_t + \sum_{l=-K}^{-2} \delta_{l} 1{E_i = e} D_{i,t}^{l} + \sum_{l=0}^{M} \delta_{l} \delta_{l} 1{E_i = e} D_{i,t}^{l} + \mathbf{X_{i, t} \gamma} + \varepsilon_{i, t}$$
	
3. Calculate the interaction-weighted average treatment effects using output from steps 1 and 2 for every relative time $l$.
	$$\hat{\beta_l} = \sum_{e} \hat{\delta_{l}} \widehat{\mathbb{E}} (E_i = e | E_i \in g ) \ forall \ l$$ 
	

### Documentation
```python
paneleventstudy.interactionweighted_eventstudy(data, outcome, event, group, cohort, reltime, calendartime, covariates, vcov_type='robust', check_balance=True)
```
#### Parameters
#### Output
### Example


## Single Entity Event Study 

Estimates dynamic treatment effects ($\beta_{l}$ coefficients on leads and lags of treatment dummies) using an single entity linear regression model with an OLS estimator, where $l=0$ refers to when the treatment was applied. 
$D_{l}$ are dummies switching when the entity is $l$ periods relative to the treatment onset.
The linear regression includes a constant ($\alpha$), is an optional vector of controls ($\mathbf{X_{t}}$).
$\epsilon_{t}$ are the errors.

$$Y_{t} = \alpha + \sum_{l=-K}^{-2} \beta_{l} D_{l} + \sum_{l=0}^{M} \beta_{l} D_{l} + \mathbf{X_{t} \gamma} + \epsilon_{t}$$

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
