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
paneleventstudy.dropmissing()
```
#### Parameters
#### Output
### Example

## Checking if input dataframe is a balanced panel
### Documentation
```python
paneleventstudy.balancepanel()
```
#### Parameters
#### Output
### Example

## Generate column indicating control groups (never-treated / last-treated)
### Documentation
```python
paneleventstudy.identifycontrols()
```
#### Parameters
#### Output
### Example

## Generate relative time column (treatment onset = 0)
### Documentation
```python
paneleventstudy.genreltime()
```
#### Parameters
#### Output
### Example

## Generate column indicating treatment cohorts
### Documentation
```python
paneleventstudy.gencohort()
```
#### Parameters
#### Output
### Example

## Generate calendar time with integers
### Documentation
```python
paneleventstudy.gencalendartime_numerics()
```
#### Parameters
#### Output
### Example

## Identify collinear or invariant columns
### Documentation
```python
paneleventstudy.checkcollinear()
```
#### Parameters
#### Output
### Example

## Identify linearly dependent columns
### Documentation
```python
paneleventstudy.checkfullrank()
```
#### Parameters
#### Output
### Example

# Implementation (Analytical)
## Naive TWFE Panel Event Study
### Documentation
```python
paneleventstudy.naivetwfe_eventstudy()
```
#### Parameters
#### Output
### Example


## Interaction-Weighted Panel Event Study 
### Documentation
```python
paneleventstudy.interactionweighted_eventstudy()
```
#### Parameters
#### Output
### Example


## Single Entity Event Study 
### Documentation
```python
```
#### Parameters
#### Output
### Example

# Implementation (Utilities)
## Plotting Event Study Lead-Leag Coefficients
### Documentation
```python
paneleventstudy.eventstudyplot()
```
#### Parameters
#### Output
### Examples

## Sending images using telegram bots
### Documentation
```python
paneleventstudy.telsendimg()
```
#### Parameters
#### Output
### Examples

## Sending files using telegram bots
### Documentation
```python
paneleventstudy.telsendfiles()
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
