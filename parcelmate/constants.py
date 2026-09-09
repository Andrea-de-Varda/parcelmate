import re

CONNECTIVITY_NAME = 'connectivity'
PARCELLATION_NAME = 'parcellation'
SUBNETWORK_NAME = 'subnetwork'
KNOCKOUT_NAME = 'knockout'
STABILITY_NAME = 'stability'
SAMPLE_NAME = 'sample'

N_SAMPLES = 4
N_TOKENS = 100000
EXTENSION = '.h5'
HALF_NAMES = ('halfA', 'halfB')  # split-half connectivity, for reliability/fidelity
INPUT_NAME_RE = re.compile(r'(%s|%s)_(.+)_(%s\d+|halfA|halfB|avg)%s' % (
    CONNECTIVITY_NAME, PARCELLATION_NAME, SAMPLE_NAME, EXTENSION)
)

# Names reserved at the top level of a run directory, so a variant cannot be called one of
# them and collide with a shared artefact.
RESERVED_VARIANT_NAMES = (CONNECTIVITY_NAME, SUBNETWORK_NAME, KNOCKOUT_NAME, 'plots', 'metrics')

OUTPUT_DIR = 'results'
PLOT_DIR = 'plots'
