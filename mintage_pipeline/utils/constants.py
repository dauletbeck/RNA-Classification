OUTPUT_FOLDER = "out/"
RELEVANT_ATOMS = [" P  ", " O5*", " C5*", " C4*", " C3*", " O3*", " O4*",
                  " C1*", " N9 ", " C4 ", " N1 ", " C2 ", " C2*", " O2*"]
RELEVANT_ATOMS_BACKBONE_HYDROGEN_ATOMS = [" H5*", "H5**", " H4*", " H3*"]
RELEVANT_RING_ATOMS = [" O4*", " C1*", " C2*", " O2*"]
RELEVANT_RING_ATOMS_HYDROGEN = [" H1*", " H2*", "HO2*"]
RELEVANT_OXYGEN_ATOMS = [" OP1", " OP2"]
SUGAR_ATOMS = [" C4*", " C3*", " C2*", " C1*", " O4*"]
RELEVANT_ATOMS_ONE_RING = [" O4*", " C1*", " N1 ", " C2 "]
RELEVANT_ATOMS_TWO_RING = [" O4*", " C1*", " N9 ", " C4 "]
ONE_RING_BASES = ["  U", "  C"]  # , "  T"] thymin not in RNA (only DNA)
TWO_RING_BASES = ["  G", "  A"]
BASES = ONE_RING_BASES + TWO_RING_BASES
RNA_BASES_VALIDATION = ["U", "C", "G", "A"]
BASE_ATOMS_VALIDATION = ['N4', 'N2', 'C6', 'H3', 'C5', 'H22', 'N7', 'C2', 'N3', 'O2', 'H41', 'N9', 'O6',
                         'H8', 'H21', 'H5', 'N1', 'H2', 'H6', 'N6', 'H62', 'C8', 'H61', 'C4', 'O4', 'H42', 'H1']
RING_ATOMS_VALIDATION = ["O4'", "C1'", "H1'", "C2'", "H2'", "O2'", "HO2'"]
BACKBONE_ATOMS_VALIDATION = ["C4'", "H4'", "C3'", "H3'", "O3'", 'P', "O5'", "C5'", "H5'", "H5''", 'OP1', 'OP2']

COLORS = ['black', 'darkred', 'darkgreen', 'darkmagenta', 'royalblue', 'grey', 'pink', 'yellow', 'teal', 'gold', 'navy',
          'magenta', 'darkviolet', 'tomato', 'peru', 'darkkhaki', 'darkslategray', 'springgreen'] + ['black'] * 1000

COLORS_SCATTER = ['black', 'darkred', 'pink', 'teal', 'green', 'grey', 'darkmagenta', 'dodgerblue', 'navy',
                  # darkgreen, royalblue
                  'gold', 'khaki', 'darkkhaki', 'mediumpurple', 'tomato', 'peru', 'springgreen', 'magenta',
                  'darkslategray', "forestgreen", "darkgreen", "orchid", "royalblue", "blueviolet",
                  "indigo", "darkorange",
                  'aliceblue', 'aqua', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blue', 'brown',
                  'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson',
                  'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkkhaki', 'darkmagenta',
                  'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
                  'darkturquoise', 'deeppink','deepskyblue', 'dimgray', 'indianred', 'indigo', 'ivory', 'lavender',
                  'lavenderblush'] + ['black'] * 1000

MARKERS = ['.', 'p', 's', '*', 'd', 'D', 'P', 'p', '^', '<', '>', 'X', 'o', 'v', '8'] + ['p'] * 1000
# MARKERS = ['o'] * 10

mean_shape_five_chain = [[-1.16818573, 5.19807613, -0.11382662],
                         [-1.86256684, 4.28693153, 0.25897307],
                         [0., 0., 0.],
                         [-0.64923387, 1.51142251, -4.93969874],
                         [0.195846320, 2.25434678, -4.75476828]]
mean_shape_suites = [[-0.12591091, 2.45877066, 3.74032015],
                     [-0.67100460, 2.34143495, 2.70605396],
                     [-0.17027422, 1.52294343, 1.76655226],
                     [-0.36251817, 0.22964374, 1.71909763],
                     [0.465046290, -0.88406471, 1.19668376],
                     [0.394358130, -0.75697418, -0.32848019],
                     [-0.36493994, -0.99902266, -1.15426669],
                     [-0.29585098, -0.6859287, -2.53959362],
                     [0.653088390, -1.10312211, -3.2834452],
                     [0.478006000, -2.12368043, -3.82292206]]
mean_shape_six_chain = [[-0.87746876, 3.12149999, 2.40706025],
                        [-1.52409606, 2.22841907, 2.73435857],
                        [0.444485120, -1.64352595, 2.28121378],
                        [-0.23381563, -0.22665184, -2.58040459],
                        [0.625835790, 0.74054131, -2.3910034],
                        [1.565059540, -4.22028257, -2.45122461]]
mean_shape_suites_shape = [[-0.01392428, 0.26566443, 0.40061165],
                           [-0.07151346, 0.25226462, 0.29074399],
                           [-0.01851063, 0.16347634, 0.19093318],
                           [-0.03868052, 0.02418145, 0.18558549],
                           [0.049255980, -0.09617092, 0.12955191],
                           [0.042181000, -0.08243948, -0.03488875],
                           [-0.03910373, -0.10777705, -0.1240591],
                           [-0.03123859, -0.07401430, -0.27339117],
                           [0.070098210, -0.11825045, -0.35334854],
                           [0.051436030, -0.22693465, -0.41173866]]
mean_shape_mesoscopic_shape = [[0.065027970, 0.36070843, 0.41816877],
                               [-0.05861579, 0.15166466, 0.33932101],
                               [-0.15478157, -0.04242084, 0.15305055],
                               [-0.13009300, -0.14976051, -0.10090082],
                               [0.032284710, -0.16786324, -0.33511186],
                               [0.246177670, -0.1523285, -0.47452765]]
mean_shape_mesoscopic = [[1.176999900, 6.95733533, 8.1505349],
                         [-1.14792905, 2.92003487, 6.56594136],
                         [-2.90090310, -0.76009351, 2.9290466],
                         [-2.42975804, -2.83290562, -1.95053775],
                         [0.607408850, -3.25957673, -6.47735654],
                         [4.694181430, -3.02479434, -9.21762858]]
mean_shapes_all = [mean_shape_five_chain, mean_shape_six_chain, mean_shape_suites,
                   mean_shape_mesoscopic, mean_shape_suites_shape, mean_shape_mesoscopic_shape]
