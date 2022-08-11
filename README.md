# paneleventstudy
This Python package implements the panel (and single entity) event study models, covering the naive two-way fixed effects implementation, and the interaction-weighted implementation from Sun and Abraham (2021) (derived from https://github.com/lsun20/EventStudyInteract).

The package includes three sets of functions:
1. ```Data cleaning```: Functions to prepare data frames for the analytical set of functions, e.g., ensuring that they are in the right format, and have the right columns (with the right content)
2. ```Analytical```: Direct implementation of the event study models
3. ```Utilities```: Tools to assist the user in setting up input-output flows
 
# Installation
1. ```pip install paneleventstudy```

# Examples

Refer to the JuPyTeR notebook ```example_paneleventstudy.ipynb```

# Implementation Notes (Data Cleaning)
## Counting and dropping missing observations
### Documentation
```python
paneleventstudy.dropmissing(data, event)
```
#### Parameters
```data```:
	[pandas dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)

```event```:
	String matching the label of the column in ```data``` corresponding to the event variable; this should be a dummy variable indicating the pre- (values 0 prior to relative time 0) and post- periods (values 1 from relative time 0 onwards)

#### Output
1. A copy of ```data``` with rows corresponding to missing ```event``` dropped
2. Display on the interface the number of rows in ```data```, and the number of rows in the output data frame


## Checking if input dataframe is a balanced panel
Most panel event study methods in the literature, and is the case at present for all methods covered in this package, only work with balanced panel data. 
That is to say that all entities in the data set must have the same number of time periods. 
This package checks if the input data is indeed a balanced panel with entities $i \in [0, 1, ..., N-1, N]$ and time periods $t \in [0, 1, 2, ..., T-1, T]$ in two steps.

1. Check if all entities $i$ have the same number of time periods
	$$L(\mathbf{t}_{i}) = L_T \ \forall \ i \ for \ L_T \in \mathcal{N}^{+} $$

2. Optionally, if the calendar time variable in the input data frame is numeric, further check if the smallest and largest time values are the same for all entities $i$
	
	$$\min(\mathbf{t}_{i}) = L_Tmin \ \forall \ i \ for \ L_Tmin \in \mathcal{N}^{0+} $$
	
	$$\max(\mathbf{t}_{i}) = L_Tmax \ \forall \ i \ for \ L_Tmax \in \mathcal{N}^{+} $$

### Documentation
```python
paneleventstudy.balancepanel(data, group, event, calendartime, check_minmax=True)
```
#### Parameters
```data```: 
	[pandas dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)

```group```:
	String matching the label of the column in ```data``` containing the categorical levels of the individual entities

```event```:
	String matching the label of the column in ```data``` corresponding to the event variable; this should be a dummy variable indicating the pre- (values 0 prior to relative time 0) and post- periods (values 1 from relative time 0 onwards)

```calendartime```:
	Integers or integers matching the label of the column in ```data``` containing calendar times going from the earliest to last time period; this can be user-fed or generated from ```gencalendartime_numericscalendartime```.

```check_minmax```:
	Boolean to trigger option for a deeper check, which verifies if all entities in ```group``` have the same minimum and maximum values in ```calendartime```; default is ```True```, and can be used when the ```calendartime``` column is generted from ```gencalendartime_numericscalendartime```, or are already preset as integers

#### Output
A Boolean indicating if ```data``` is balanced.


## Generate column indicating control groups (never-treated / last-treated)
In the difference in difference (DiD) methodology, which event studies are a variant of, if treatment is truly exogenous, the treatment effect is estimated by comparing the average outcomes of the treated group (received treatment) against the control group (did not receive treatment).

Discussing endogeneity aside, in panel event studies, and indeed dynamic DiD, it is possible for no groups to be never-treated, e.g., in staggered DiD setups. 
Choosing the right control group is essential in establishing the right counterfactual, on which unbiased or consistent treatment effect estimates are conditioned on.
In these cases, we may want to use the last-treated group as a control group. This was argued prominently in recent DiD papers, such as [Callaway and Sant'Anna](https://doi.org/10.1016/j.jeconom.2020.12.001) and [Sun and Abraham (2021)](https://doi.org/10.1016/j.jeconom.2020.09.006).

This function tells us which group(s) is / are the control groups, whether never-treated or last-treated, which is essential for the analytical functions later.

### Documentation
```python
paneleventstudy.identifycontrols(data, group, event)
```
#### Parameters
```data```: 
	[pandas dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)

```group```:
	String matching the label of the column in ```data``` containing the categorical levels of the individual entities

```event```:
	String matching the label of the column in ```data``` corresponding to the event variable; this should be a dummy variable indicating the pre- (values 0 prior to relative time 0) and post- periods (values 1 from relative time 0 onwards)

#### Output
A copy of ```data``` with a new column labelled ```control_group``` indicating if the entity in ```group``` is a control group (never-treated or last-treated).



## Generate relative time column (treatment onset = 0)
Event studies methodologies essentially estimate the dynamic treatment effect relative to the onset of treatment (i.e., before and after treatment). 
This is akin to asking "what is the effect of treatment $D$ on outcome $Y$ at different timepoints relative to the treatment timing?".

This function generates a column containing these relative times from two sets of information:
1. Calendar time; and 
2. When the treatment or event happened

### Documentation
```python
paneleventstudy.genreltime(data, group, event, calendartime, reltime='reltime', check_balance=True)
```
#### Parameters
```data```: 
	[pandas dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)

```group```:
	String matching the label of the column in ```data``` containing the categorical levels of the individual entities

```event```:
	String matching the label of the column in ```data``` corresponding to the event variable; this should be a dummy variable indicating the pre- (values 0 prior to relative time 0) and post- periods (values 1 from relative time 0 onwards)

```calendartime```:
	Integers matching the label of the column in ```data``` containing calendar times going from 0 (earliest time period) to T (last time period); this can be generated from ```gencalendartime_numericscalendartime```

```reltime```:
	String to be used as the label of a new column containing relative times going from -L to +K as integers, with 0 being the timing of treatment onset

```check_balance```:
	Checks if ```data``` is a balanced panel; default option is ```True```

#### Output
A copy of ```data``` with a new column labelled ```reltime``` containing the relative times for all calendar times in ```calendar``` by entities in ```group```.



## Generate column indicating treatment cohorts
[Sun and Abraham (2021)](https://doi.org/10.1016/j.jeconom.2020.09.006)'s interaction-weighted event study methodology requires (1) the estimation of cohort-specific treatment effects, and (2) cohort shares by relative times.
To do this, the methodology requires an identifier for groups that were treated in the same calendar time. 

### Documentation
```python
paneleventstudy.gencohort(data, group, event, calendartime, cohort='cohort', check_balance=True)
```
#### Parameters
```data```: 
	[pandas dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)

```group```:
	String matching the label of the column in ```data``` containing the categorical levels of the individual entities

```event```:
	String matching the label of the column in ```data``` corresponding to the event variable; this should be a dummy variable indicating the pre- (values 0 prior to relative time 0) and post- periods (values 1 from relative time 0 onwards)

```calendartime```:
	Integers matching the label of the column in ```data``` containing calendar times going from 0 (earliest time period) to T (last time period); this can be generated from ```gencalendartime_numericscalendartime```

```cohort```:
	String to be used as the label of a new column containing the treatment cohort of respective entities in ```group```; default is ```'cohort'```

```check_balance```:
	Checks if ```data``` is a balanced panel; default option is ```True```

#### Output
A copy of ```data``` with a new column labelled ```cohort``` indicating the treatment cohort that the entities ```group``` belong to.



## Generate calendar time with integers
For generalise across the infinitely many possible formats that calendar times can be presented in (e.g., miliseconds, seconds, days, months, quarters, years, or even custom ones), calendar times can be converted into numerics.
This eases computation in the rest of the package, by converting the calendar time column into integers starting from 0 (earliest) to T (latest).

### Documentation
```python
paneleventstudy.gencalendartime_numerics(data, group, event, calendartime, calendartime_numerics='ct')
```
#### Parameters
```data```: 
	[pandas dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)

```group```:
	String matching the label of the column in ```data``` containing the categorical levels of the individual entities

```event```:
	String matching the label of the column in ```data``` corresponding to the event variable; this should be a dummy variable indicating the pre- (values 0 prior to relative time 0) and post- periods (values 1 from relative time 0 onwards)

```calendartime```:
	Column matching the label of the column in ```data``` containing calendar times going from the earliest time period to the last time period

```calendartime_numerics```:
	String to be used as the label of a new column containing the calendar times converted into nonnegative integers with 0 being the earliest, and T being the latest period

#### Output
A copy of ```data``` with a new column labelled ```calendartime_numerics``` with numeric version of ```calendartime```, which can then be passed to the analytical functions.



## Identify collinear or invariant columns
The basic functional form of estimating equations in the DiD and event study methodology is a linear regression, which requires variables in the RHS of the equation to not be multicollinear, or invariant.
This function checks if this is indeed the case.

### Documentation
```python
paneleventstudy.checkcollinear(data, rhs)
```
#### Parameters
```data```: 
	[pandas dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)

```rhs```:
	A list containing strings matching the labels of the columns in ```data``` to be checked for collinearity and invariance; precedence goes to columns in the rightmost of ```rhs``` (if two columns are collinear, the one appearing later in ```rhs``` is not included in the output)

#### Output
A list of labels in ```rhs``` which should be dropped to avoid multicollinearity or invariance in ```rhs``` columns in ```data```.



## Identify linearly dependent columns
The basic functional form of estimating equations in the DiD and event study methodology is a linear regression, which requires the matrix containing the variables in the RHS of the equation to satisfy full column rank.
This function checks if this is indeed the case.

### Documentation
```python
paneleventstudy.checkfullrank(data, rhs, intercept='Intercept')
```
#### Parameters
```data```: 
	[pandas dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)

```rhs```:
	A list containing strings matching the labels of the columns in ```data``` to be checked for full rank; precedence goes to columns in the rightmost of ```rhs```

```intercept```:
	String containing the label of the intercept column (column of numerics 1), which will be given precedence in the procedure; set as ```None``` if no intercepts are contained in ```data```, and the default is ```'Intercept'```, which is the default when using [patsy.dmatrices()](https://patsy.readthedocs.io/en/latest/API-reference.html)

#### Output
A list of labels in ```rhs``` which should be dropped to for the matrix containing ```rhs``` columns in ```data``` to satisfy full rank.



# Implementation Notes (Analytical)
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
```data```: 
	[pandas dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)

```outcome```: 
	String matching the label of the column in ```data``` corresponding to the outcome variable; this is the LHS variable in the regression

```event```:
	String matching the label of the column in ```data``` corresponding to the event variable; this should be a dummy variable indicating the pre- (values 0 prior to relative time 0) and post- periods (values 1 from relative time 0 onwards)

```group```:
	String matching the label of the column in ```data``` containing the categorical levels of the individual entities
	
```reltime```:
	Integers matching the label of the column in ```data``` containing relative times going from -L to +K, with 0 being the timing of treatment onset; this can be generated from ```calendartime``` generated from ```genreltime```, and ```reltime=-1``` is automatically chosen as the reference period

```calendartime```:
	Integers matching the label of the column in ```data``` containing calendar times going from 0 (earliest time period) to T (last time period); this can be generated from ```gencalendartime_numericscalendartime```.

```covariates```:
	List of columns corresponding to control variables in ```data``` to be included in the RHS of the regression; if no covariates are to be included, set ```covariates=[]```

```vcov_type```:
	String corresponding to the type of variance-covariance estimator in [linearmodels.PanelOLS.fit()](https://bashtage.github.io/linearmodels/panel/panel/linearmodels.panel.model.PanelOLS.fit.html), which is called during the estimation process; default option is ```'robust'```
	
```check_balance```:
	Checks if ```data``` is a balanced panel; default option is ```True```

#### Output
Returns a [pandas dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) with 3 columns, indexed to ```reltime```: 
1. ```parameter```: The point estimates of the interaction-weighted average treatment affects
2. ```lower```: The lower confidence bound of ```parameter```
3. ```upper```:  The upper confidence bound of ```parameter```



## Interaction-Weighted Panel Event Study 
Estimates dynamic treatment effects using the interaction-weighted estimator described in [Sun and Abraham (2021)](https://doi.org/10.1016/j.jeconom.2020.09.006).
Again, for the following structural equation, we are interested in estimating $\beta_{l}$, the coefficients on leads and lags of treatment dummies, where $l$ is relative time as in [Sun and Abraham (2021)](https://doi.org/10.1016/j.jeconom.2020.09.006), i.e., the time period relative to treatment onset. 
$l=0$ refers to when the treatment was applied to entity $i$.

$$Y_{i,t} = \alpha_i + \alpha_t + \sum_{l=-K}^{-2} \beta_{l} D_{i, t}^{l} + \sum_{l=0}^{M} \beta_{l} D_{i, t}^{l} + \mathbf{X_{i, t} \gamma} + \eta_{i, t}$$

This implementation has 3 broad steps.

1. Calculate the cohort shares by relative time, $\mathbb{E} (E_i = e | E_i \in g )$ where $g$ is the set of relative times included in the analysis. This package uses a no-constant linear regression model with an OLS estimator as per the Sun and Abraham (2021)'s original Stata package [here](https://github.com/lsun20/EventStudyInteract). Using a linear regression approach, instead of simple tabulation, allows for calculation of standard errors of the cohort share estimates.
	
	$$1\{E_i = e | E_i \in g \} = w_{e,l} D_{i, t}^{l} + e_i$$
	
2. Estimate the cohort-specific average treatment effects, $CATT_{e, l}$, by interacting the cohort dummy with the treatment / relative time dummy, $1(E_i = e) D_{i,t}^{l}$.
	
	$$Y_{i,t} = \alpha_i + \alpha_t + \sum_{l=-K}^{-2} \delta_{l} 1(E_i = e) D_{i,t}^{l} + \sum_{l=0}^{M} \delta_{l} 1(E_i = e) D_{i,t}^{l} + \mathbf{X_{i, t} \gamma} + \varepsilon_{i, t}$$
	
3. Calculate the interaction-weighted average treatment effects using output from steps 1 and 2 for every relative time $l$. In this current version, the estimated confidence bands are scaled the same way.
	
	$$\hat{\beta_l} = \sum_{e} \hat{\delta_{l}} \hat{w_{e,l}} \ \forall \ l$$ 
	

### Documentation
```python
paneleventstudy.interactionweighted_eventstudy(data, outcome, event, group, cohort, reltime, calendartime, covariates, vcov_type='robust', check_balance=True)
```
#### Parameters

```data```: 
	[pandas dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)

```outcome```: 
	String matching the label of the column in ```data``` corresponding to the outcome variable; this is the LHS variable in the regression

```event```:
	String matching the label of the column in ```data``` corresponding to the event variable; this should be a dummy variable indicating the pre- (values 0 prior to relative time 0) and post- periods (values 1 from relative time 0 onwards)

```group```:
	String matching the label of the column in ```data``` containing the categorical levels of the individual entities
	
```cohort```:
	Integers matching the label of the column in ```data``` containing the categorical levels of the cohorts in the data set generated from ```gencohort``` (e.g., all entities treated in calendar time 3 should take the value 3 in this column)

```reltime```:
	Integers matching the label of the column in ```data``` containing relative times going from -L to +K, with 0 being the timing of treatment onset; this can be generated from ```calendartime``` generated from ```genreltime```, and ```reltime=-1``` is automatically chosen as the reference period

```calendartime```:
	Integers matching the label of the column in ```data``` containing calendar times going from 0 (earliest time period) to T (last time period); this can be generated from ```gencalendartime_numericscalendartime```.

```covariates```:
	List of columns corresponding to control variables in ```data``` to be included in the RHS of the regression; if no covariates are to be included, set ```covariates=[]```

```vcov_type```:
	String corresponding to the type of variance-covariance estimator in [linearmodels.PanelOLS.fit()](https://bashtage.github.io/linearmodels/panel/panel/linearmodels.panel.model.PanelOLS.fit.html), which is called during the estimation process; default option is ```'robust'```
	
```check_balance```:
	Checks if ```data``` is a balanced panel; default option is ```True```

#### Output
Returns a [pandas dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) with 3 columns, indexed to ```reltime```: 
1. ```parameter```: The point estimates of the interaction-weighted average treatment affects
2. ```lower```: The lower confidence bound of ```parameter```
3. ```upper```:  The upper confidence bound of ```parameter```



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
```data```: 
	[pandas dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)

```outcome```: 
	String matching the label of the column in ```data``` corresponding to the outcome variable; this is the LHS variable in the regression

```reltime```:
	Integers matching the label of the column in ```data``` containing relative times going from -L to +K, with 0 being the timing of treatment onset

```covariates```:
	List of columns corresponding to control variables in ```data``` to be included in the RHS of the regression; if no covariates are to be included, set ```covariates=[]```

```vcov_type```:
	String corresponding to the type of variance-covariance estimator in [statsmodels.regression.linear_model.RegressionResults.get_robustcov_results](https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html#statsmodels.regression.linear_model.RegressionResults.get_robustcov_results), which is called during the estimation process; default option is ```'HC3'```
	

#### Output
Returns a [pandas dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) with 3 columns, indexed to ```reltime```: 
1. ```parameter```: The point estimates of the interaction-weighted average treatment affects
2. ```lower```: The lower confidence bound of ```parameter```
3. ```upper```:  The upper confidence bound of ```parameter```


# Implementation Notes (Utilities)
## Plotting Event Study Lead and Lag Coefficients
This function calls plotly's [graph_objects](https://plotly.com/python/graph-objects/) module to show the event study estimates (dynamic treatment effects) to be shown as a line chart, together with their confidence bands (can be manually excluded).
Moreover, it exports an interactive graph as a html file via plotly's [plotly.io.write_html()](), and a static graph as a png file via plotly's [plotly.io.write_image()].
Users of this package may, of course, opt to other charting packages, modules, or scripts to plot the event study estimates.

### Documentation
```python
paneleventstudy.eventstudyplot(input, big_title='Event Study Plot (With 95% CIs)', path_output='', name_output='eventstudyplot')
```
#### Parameters
```input```:
	Output from the analytical functions (```paneleventstudy.naivetwfe_eventstudy()```, ```paneleventstudy.interactionweighted_eventstudy()```, ```paneleventstudy.timeseries_eventstudy()```); manually exclude columns ```lower``` and ```upper``` if only the point estimates are to be shown

```big_title```:
	String containing the main title of the figure; default is ```'Event Study Plot (With 95% CIs)'```

```path_output```:
	String containing the directory of where the output files should be saved in; default is ```''```, i.e., the present working directory

```name_output```=
	String containing the file name of the image and html file to be generated; default is ```'eventstudyplot'```

#### Output


# Requirements
## Python Packages
- pandas>=1.4.3
- numpy>=1.23.0
- linearmodels>=4.27
- plotly>=5.9.0
- statsmodels>=0.13.2
- sympy>=1.10.1
