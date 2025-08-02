"""
RNA structural constants and atom definitions.

Moved from mintage/utils/constants.py with only the parts relevant 
to low-resolution analysis.
"""

# Relevant atoms for RNA structure analysis
RELEVANT_ATOMS = [" P  ", " O5*", " C5*", " C4*", " C3*", " O3*", " O4*",
                  " C1*", " N9 ", " C4 ", " N1 ", " C2 ", " C2*", " O2*"]

RELEVANT_ATOMS_BACKBONE_HYDROGEN_ATOMS = [" H5*", "H5**", " H4*", " H3*"]
RELEVANT_RING_ATOMS = [" O4*", " C1*", " C2*", " O2*"]
RELEVANT_RING_ATOMS_HYDROGEN = [" H1*", " H2*", "HO2*"]
RELEVANT_OXYGEN_ATOMS = [" OP1", " OP2"]
SUGAR_ATOMS = [" C4*", " C3*", " C2*", " C1*", " O4*"]

# Base atom definitions
RELEVANT_ATOMS_ONE_RING = [" O4*", " C1*", " N1 ", " C2 "]
RELEVANT_ATOMS_TWO_RING = [" O4*", " C1*", " N9 ", " C4 "]

# RNA bases
ONE_RING_BASES = ["  U", "  C"]
TWO_RING_BASES = ["  G", "  A"]
BASES = ONE_RING_BASES + TWO_RING_BASES

# Plotting colors and markers
COLORS_SCATTER = ['black', 'darkred', 'pink', 'teal', 'green', 'grey', 'darkmagenta', 'dodgerblue', 'navy',
                  'gold', 'khaki', 'darkkhaki', 'mediumpurple', 'tomato', 'peru', 'springgreen', 'magenta',
                  'darkslategray', "forestgreen", "darkgreen", "orchid", "royalblue", "blueviolet",
                  "indigo", "darkorange"] + ['black'] * 1000

MARKERS = ['.', 'p', 's', '*', 'd', 'D', 'P', 'p', '^', '<', '>', 'X', 'o', 'v', '8'] + ['p'] * 1000

# Low-resolution coordinate labels
LOW_RES_LABELS = [r'$d_2$', r'$d_3$', r'$\alpha$', r'$\theta_1$', r'$\phi_1$', r'$\theta_2$', r'$\phi_2$']

# Coordinate ranges for plotting
LOW_RES_RANGES = [[3,5.5], [4,6.5], [0,180], [0,180], [-180,180], [0,180], [-180,180]]