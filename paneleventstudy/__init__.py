from importlib import resources

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# Version of the package
__version__ = "0.0.0"

# Load scripts / classes / functions so that they can be called directly

from dataprep import (
dropmissing,
balancepanel,
identifycontrols,
genreltime,
gencohort,
gencalendartime_numerics,
checkcollinear,
checkfullrank,
)

from naivetwfe_eventstudy import (
naivetwfe_eventstudy,
)

from interactionweighted_eventstudy import (
interactionweighted_eventstudy,
)

from eventstudyplot import (
eventstudyplot,
)