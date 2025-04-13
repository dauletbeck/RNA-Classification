# -*- coding: utf-8 -*-
"""
Copyright (C) 2014-2015 Benjamin Eltzner

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 3
of the License, or (at your option) any later version.

This software is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public
License along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA
or see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import numpy.linalg as la
import xml.etree.ElementTree as ET
import re, os, fnmatch
from PNDS_tree import Node
from math import pi, sqrt, atan2

################################################################################
################################   Constants   #################################
################################################################################

OUTPUT_FOLDER = "./out/Torus_PCA/"
RELEVANT_ATOMS = [" P  ", " O5*", " C5*", " C4*", " C3*", " O3*", " O4*",
                  " C1*", " N9 ", " C4 ", " N1 ", " C2 ", " C2*", " O2*"]
ONE_RING_BASES = ["  U", "  C", "  T"]
TWO_RING_BASES = ["  G", "  A"]
BASES = ONE_RING_BASES + TWO_RING_BASES
T_RNAS = ["1b23", "1ehz", "1f7u", "1ffy", "1h3e", "1h4s", "1ivs", "1n78",
          "1qf6", "1qtq", "1vfg", "1yfg", "2bte", "2csx", "2fmt"]
T_RNA_TT = ["1h3e", "1h4s", "1ivs", "1n78", "2bte"]
T_RNA_EC = ["1b23", "1qf6", "1qtq", "2fmt"]
RIBOSOMES = ["1s72", "1xmq"]
INTRONS = ["1hr2", "1kxk"]
AXES = np.array([[sqrt(0.5), 0, 0, 0, -sqrt(0.5), 0],
                 [0, 1/sqrt(6), 0, 1/sqrt(6), 0, -2/sqrt(6)],
                 [-1/sqrt(6), 0, 2/sqrt(6), 0, -1/sqrt(6), 0]])

################################################################################
##############################   I/O Functions   ###############################
################################################################################

def import_csv (filename):
    data = __get_raw_data(filename, as_type='Float')
    for key in data:
        tmp = np.array(discard_empty_lists(data[key]))
        data[key] = np.squeeze(tmp)
    if len(data) == 1 and 'NO_LABEL' in data:
        return data['NO_LABEL']
    return data

def import_lists (filename, as_type='Int'):
    data = __get_raw_data(filename, as_type=as_type)
    for key in data:
        data[key] = discard_empty_lists(data[key])
        if len(data[key]) == 1:
            data[key] = data[key][0]
    if len(data) == 1 and 'NO_LABEL' in data:
        return data['NO_LABEL']
    return data

def __get_raw_data (filename, as_type='Int'):
    data = {}
    label = None
    depth = 0
    with open(filename) as datafile:
        for line in datafile:
            if line[0] == '%':
                label = line[1:].strip()
                data[label] = []
                depth = 0
            elif line[0] != '#' and line[0] != ',':
                if label == None:
                    label = 'NO_LABEL'
                    data[label] = []
                level = data[label]
                for i in range(depth):
                    level = level[-1]
                line = line.replace("NA", "nan")
                tmp = filter(None, re.split(',| |\t', line.strip()))
                if as_type == 'Int':
                    level.append([int(i) for i in tmp])
                elif as_type == 'Float':
                    level.append([float(i) for i in tmp])
                else:
                    level.append(list(tmp))
            elif line[0] == ',' and label != None:
                tmp = 0
                while ','*(tmp+2) == line[:tmp+2]:
                    tmp += 1
                if tmp == 0:
                    continue
                for i in range(tmp - depth):
                    data[label] = [data[label]]
                    depth += 1
                level = data[label]
                for i in range(depth - tmp):
                    level = level[-1]
                for i in range(tmp):
                    level.append([])
                    level = level[-1]
    return data

def import_xml (filename, point_data):
    root = ET.parse(filename).getroot()
    tree = []
    return xml_recursion(root, point_data, tree)[1], tree

def xml_recursion (node, point_data, tree):
    child1 = None; child2 = None
    for subnode in node:
        tag, sub = xml_recursion(subnode, point_data, tree)
        if tag == 0:
            child1 = sub
        elif tag == 1:
            child2 = sub
        else:
            print('UNKNOWN TAG:', tag)
    d = node.attrib; point = None
    if 'point' in d:
        point = point_data[int(d['point'])]
    this = Node(point, child1, child2, int(d['level']), int(d['size']))
    tree.append(this)
    return int(node.tag.replace('cluster','0')) % 10, this

def import_from_pdb (filename, dict_values, verbose=False):
    atom_dict, residue_dict, residue_types, head_residues, tail_residues = __parse_pdb(filename, verbose)
    for key in dict_values:
        all_angles = []
        for this in dict_values[key]:
            if this in residue_dict:
                this = residue_dict[this]
            else:
                print('Not in dict:', filename, key, this)
            error_msg = filename + ' ' + key + ' ' + str(this)
            angles = __calc_angles(atom_dict, residue_types, this, verbose)
            if not None in angles:
                all_angles.append(angles)
            elif verbose: print('Residue skipped!', error_msg)
        dict_values[key] = all_angles
    return dict_values

def import_pdb_file (filename, verbose=False):
    print(filename)
    atom_dict, _, residue_types, head_residues, tail_residues = __parse_pdb(filename, verbose)
    all_angles = []
    for i, head in enumerate(head_residues):
        total_residues = tail_residues[i] - head + 1
        for j in range(1, total_residues):
            this = j + head - 1
            error_msg = filename + ' ' +  str(i) + ' ' + str(j)
            angles = __calc_angles(atom_dict, residue_types, this, verbose)
            if not None in angles:
                all_angles.append(angles)
            elif verbose: print('Residue skipped!', error_msg)
    if len(all_angles) < 1:
        print('WARNING! NO RNA FOUND IN', filename)
    print(len(all_angles))
    return np.array(all_angles)

def import_pdb_file_richardson (filename, verbose=False):
    print(filename)
    atom_dict, _, residue_types, head_residues, tail_residues = __parse_pdb(filename, verbose)
    all_angles = []
    for i, head in enumerate(head_residues):
        total_residues = tail_residues[i] - head + 1
        for j in range(1, total_residues):
            this = j + head - 1
            error_msg = filename + ' ' +  str(i) + ' ' + str(j)
            angles = __calc_alt_angles(atom_dict, residue_types, this, verbose)
            if not None in angles:
                all_angles.append(angles)
            elif verbose: print('Residue skipped!', error_msg)
    if len(all_angles) < 1:
        print('WARNING! NO RNA FOUND IN', filename)
    print(len(all_angles))
    return np.array(all_angles)

def import_chains_from_pdb_file (filename, verbose=False):
    print(filename)
    atom_dict, _, residue_types, head_residues, tail_residues = __parse_pdb(filename, verbose)
    all_angles = []
    for i, head in enumerate(head_residues):
        these_angles = []
        total_residues = tail_residues[i] - head + 1
        for j in range(1, total_residues):
            this = j + head - 1
            error_msg = filename + ' ' +  str(i) + ' ' + str(j)
            angles = __calc_angles(atom_dict, residue_types, this, verbose)
            if not None in angles:
                these_angles.append(angles)
            elif verbose: print('Residue skipped!', error_msg)
        if len(these_angles) > 0:
            all_angles.append(np.array(these_angles))
    if len(all_angles) < 1:
        print('WARNING! NO RNA FOUND IN', filename)
    return all_angles

def import_pdb_file_extended (filename, verbose=False):
    print(filename)
    atom_dict, _, residue_types, head_residues, tail_residues = __parse_pdb(filename, verbose)
    all_angles = []
    for i, head in enumerate(head_residues):
        total_residues = tail_residues[i] - head + 1
        for j in range(1, total_residues):
            this = j + head - 1
            error_msg = filename + ' ' +  str(i) + ' ' + str(j)
            angles = __calc_angles_extended(atom_dict, residue_types, this, verbose)
            if not angles is None:
                all_angles.append(angles)
            elif verbose: print('Residue skipped!', error_msg)
    if len(all_angles) < 1:
        print('WARNING! NO RNA FOUND IN', filename)
    print(len(all_angles))
    return np.array(all_angles)

def import_pdb_file_to_S2 (filename, verbose=False):
    print(filename)
    atom_dict, _, residue_types, head_residues, tail_residues = __parse_pdb(filename, verbose)
    all_triangles = []
    for i, head in enumerate(head_residues):
        total_residues = tail_residues[i] - head + 1
        for j in range(1, total_residues):
            this = j + head - 1
            error_msg = filename + ' ' +  str(i) + ' ' + str(j)
            triangle = __calc_S2(atom_dict, residue_types, this, verbose)
            if not triangle is None:
                all_triangles.append(triangle)
            elif verbose: print('Residue skipped!', error_msg)
    if len(all_triangles) < 1:
        print('WARNING! NO RNA FOUND IN', filename)
    print(len(all_triangles))
    return np.array(all_triangles)

# TODO
def import_pdb_file_to_hypersphere (filename, verbose=False):
    print(filename)
    atom_dict, _, residue_types, head_residues, tail_residues = __parse_pdb(filename, verbose)
    all_shapes = []
    for i, head in enumerate(head_residues):
        total_residues = tail_residues[i] - head + 1
        for j in range(1, total_residues):
            this = j + head - 1
            error_msg = filename + ' ' +  str(i) + ' ' + str(j)
            shape = __calc_hypersphere(atom_dict, residue_types, this, verbose)
            if not shape is None:
                all_shapes.append(shape)
            elif verbose: print('Residue skipped!', error_msg)
    if len(all_shapes) < 1:
        print('WARNING! NO RNA FOUND IN', filename)
    print(len(all_shapes))
    return np.array(all_shapes)

def __parse_pdb (filename, verbose):
    residues = {}
    chains = {}
    residue_types = {}
    head_residues = []
    tail_residues = []
    atom_dict = {name : {} for name in RELEVANT_ATOMS}
    residue_dict = {}
    atom_type = ''
    n_atom = 0
    n_residue = 0
    no_chain = True
    with open(filename) as datafile:
        for line in datafile:
            if (line[:4] != 'ATOM' and line[:6] != 'HETATM'):
                continue
            if not line[17:20] in BASES:
                if verbose: print('Unknown base type: ' + line[17:20])
            type_test = line[12:16].replace("'", "*")
            residue_test = line[21:27]
            if (type_test == atom_type and n_atom in residues and
                residues[n_atom] in residue_test):
                if verbose: print('Skipped duplicate:', line[:27])
                continue
            n_atom += 1
            atom_type = type_test
            residues[n_atom] = residue_test[:-1]
            chains[n_atom] = residue_test[:1]
            if (n_atom - 1 not in residues or
                residues[n_atom] != residues[n_atom - 1]):
                n_residue += 1
                residue_dict[int(residues[n_atom][1:])] = n_residue
                residue_types[n_residue] = line[17:20]
            if atom_type in RELEVANT_ATOMS:
                atom_dict[atom_type][n_residue] = [float(line[30:38]),
                                                   float(line[38:46]),
                                                   float(line[46:54])]
            if no_chain:
                head_residues.append(n_residue)
                no_chain = False
            elif chains[n_atom] != chains[n_atom-1]:
                head_residues.append(n_residue)
                tail_residues.append(n_residue - 1)
    tail_residues.append(n_residue)
    return atom_dict, residue_dict, residue_types, head_residues, tail_residues

def __calc_angles (atom_dict, residue_types, this, verbose):
    rna = atom_dict[' O2*']
    angles = [0] * 10
    if (((this - 1) not in rna) or (this not in rna) or
        ((this + 1) not in rna) or rna[this] == 0 or
        rna[this - 1] == 0 or rna[this + 1] == 0):
        return [None]
    else:
        tmp = []
        try:
            if residue_types[this] in TWO_RING_BASES:
                # Chi : O4* - C1* - N9 - C4
                tmp = [atom_dict[a][this] for a in RELEVANT_ATOMS[8:10]]
            elif residue_types[this] in ONE_RING_BASES:
                # Chi : O4* - C1* - N1 - C2
                tmp = [atom_dict[a][this] for a in RELEVANT_ATOMS[10:12]]
            else:
                print("Skipping unknown nucleic base:", residue_types[this])
                return [None]
            coords = ([atom_dict[a][this-1] for a in [" C4*"," O3*"]] +
                      [atom_dict[a][this] for a in RELEVANT_ATOMS[:6]] +
                      [atom_dict[a][this+1] for a in [" P  "," O5*"," C4*"]]+
                      [atom_dict[a][this] for a in RELEVANT_ATOMS[6:8]] + tmp)
            coords.append(atom_dict[" C2*"][this])
        except:
            if verbose: print('Atom not found!')
            return [None]
        # Alpha - Zeta
        angles[:6] = [dihedral(coords[i+1:i+5], verbose) for i in range(6)]
        # Chi
        angles[6] = dihedral(coords[11:15], verbose)
        # Eta
        angles[7] = dihedral([coords[0], coords[2], coords[5], coords[8]], verbose, True)
        # Theta
        angles[8] = dihedral([coords[2], coords[5], coords[8], coords[10]], verbose, True)
        # Nu2 :   C1* - C2* - C3* - C4*
        angles[9] = dihedral([coords[12], coords[15], coords[6], coords[5]], verbose, True)
    return angles

def __calc_alt_angles (atom_dict, residue_types, this, verbose):
    rna = atom_dict[' O2*']
    angles = [0] * 7
    if ((this not in rna) or ((this + 1) not in rna) or
        rna[this] == 0 or rna[this + 1] == 0):
        return [None]
    else:
        try:
            if not residue_types[this] in TWO_RING_BASES + ONE_RING_BASES:
                print("Skipping unknown nucleic base:", residue_types[this])
                return [None]
            coords = ([atom_dict[a][this] for a in RELEVANT_ATOMS[2:6]] +
                      [atom_dict[a][this+1] for a in RELEVANT_ATOMS[:6]])
        except:
            if verbose: print('Atom not found!')
            return [None]
        # Delta - Delta
        angles = [dihedral(coords[i:i+4], verbose) for i in range(7)]
    return angles

def __calc_angles_extended (atom_dict, residue_types, this, verbose):
    rna = atom_dict[' O2*']
    angles = np.zeros((6,2))
    if (((this - 1) not in rna) or (this not in rna) or
        ((this + 1) not in rna) or rna[this] == 0 or
        rna[this - 1] == 0 or rna[this + 1] == 0):
        return None
    else:
        tmp = []
        try:
            if residue_types[this] in TWO_RING_BASES:
                # Chi : O4* - C1* - N9 - C4
                tmp = [atom_dict[a][this] for a in RELEVANT_ATOMS[8:10]]
            elif residue_types[this] in ONE_RING_BASES:
                # Chi : O4* - C1* - N1 - C2
                tmp = [atom_dict[a][this] for a in RELEVANT_ATOMS[10:12]]
            else:
                print("Skipping unknown nucleic base:", residue_types[this])
                return None
            coords = ([atom_dict[a][this-1] for a in [" C4*"," O3*"]] +
                      [atom_dict[a][this] for a in RELEVANT_ATOMS[:6]] +
                      [atom_dict[a][this+1] for a in [" P  "," O5*"," C4*"]]+
                      [atom_dict[a][this] for a in RELEVANT_ATOMS[6:8]] + tmp)
            coords.append(atom_dict[" C2*"][this])
        except:
            if verbose: print('Atom not found!')
            return None
        # Alpha - Zeta
        tmp = [dihedral(coords[i+1:i+5], verbose) for i in range(6)]
        if None in tmp:
            return None
        angles[:,0] = np.array(tmp)
        coords = np.array(coords)
        diffs = coords[1:] - coords[:-1]
        diffs /= la.norm(diffs, axis=1)[:,np.newaxis]
        angles[:,1] = np.arccos(np.einsum('ij,ij->i', diffs[2:8], diffs[3:9])) * (180/pi)
        if np.any(np.isnan(angles)):
            print('Nan encountered!')
    return angles

def __calc_S2 (atom_dict, residue_types, this, verbose):
    rna = atom_dict[' O2*']
    if (((this - 1) not in rna) or (this not in rna) or
        ((this + 1) not in rna) or rna[this] == 0 or
        rna[this - 1] == 0 or rna[this + 1] == 0):
        return None
    else:
        try:
            if residue_types[this] in TWO_RING_BASES:
                # Chi : O4* - C1* - N9 - C4
                tmp = atom_dict[" N9 "][this]
            elif residue_types[this] in ONE_RING_BASES:
                # Chi : O4* - C1* - N1 - C2
                tmp = atom_dict[" N1 "][this]
            else:
                print("Skipping unknown nucleic base:", residue_types[this])
                return None
            coords = np.array([atom_dict[" P  "][this], atom_dict[" C1*"][this],
                               atom_dict[" P  "][this+1], tmp])
        except:
            if verbose: print('Atom not found!')
            return None
        # Alpha - Zeta
        triangle, rot = normalize_plane(coords[:3])
        if rot is None:
            return None
        direction = np.einsum('ij,j->i', rot, coords[-1] - coords[1])
    return direction / la.norm(direction)

# TODO
def __calc_hypersphere (atom_dict, residue_types, this, verbose):
    rna = atom_dict[' O2*']
    if (((this - 1) not in rna) or (this not in rna) or
        ((this + 1) not in rna) or rna[this] == 0 or
        rna[this - 1] == 0 or rna[this + 1] == 0):
        return None
    else:
        try:
            if not residue_types[this] in ONE_RING_BASES + TWO_RING_BASES:
                print("Skipping unknown nucleic base:", residue_types[this])
                return None
            coords = ([atom_dict[a][this] for a in RELEVANT_ATOMS[:6]] +
                      [atom_dict[" P  "][this+1]])
        except:
            if verbose: print('Atom not found!')
            return None
        b = [__diff(coords[i], coords[i+1]) for i in range(len(coords)-1)]
        for bi in b:
            bi=__dot(bi,bi)
            if bi > 3:
                if verbose: print('Atoms too far apart:', bi)
                return None
        return normalize_7(np.array(coords))
    return None

def dihedral (point_list, verbose, long=False):
    b = [__diff(point_list[i], point_list[i+1]) for i in range(3)]
    for bi in b:
        bi=__dot(bi,bi)
        if bi>(20 if long else 3):
            if verbose: print('Atoms too far apart:', bi)
            return None
    c = [__x(b[i], b[i+1]) for i in range(2)]
    tmp = atan2(__dot(__x(c[0],c[1]),__n(b[1])),__dot(c[0],c[1]))
    return (360 - tmp*(180/pi)) % 360

def normalize_plane (points):
    shift = np.mean(points, axis=0)
    p = points - shift.reshape((1,-1))
    d = np.array([(p[i] - p[i-1]) for i in range(3)])
    l = la.norm(d, axis=1).reshape((-1,1))
    if np.max(l) > 8:
        return None, None
    d /= l
    normal = np.array(__x(d[0], d[1]) if (__dot(d[1], d[2]) == 1) else __x(d[1], d[2]))
    rotation = __rotation(normal, np.array([0, 0, 1]))
    p2 = np.einsum('ij,kj->ki', rotation, p)
    rot2 = __rotation(p2[1] - p2[0], np.array([1, 0, 0]))
    rotation = np.einsum('ij,jk->ik', rot2, rotation)
    p = np.einsum('ij,kj->ki', rotation, p)[:,:2].flatten()
    return np.einsum('ij,j->i', AXES, p/la.norm(p)), rotation

def normalize_7 (points):
    points = points - points[0]
    points = points / la.norm(points[-1])
    rotation = __rotation(points[-1], np.array([1, 0, 0]))
    points = np.einsum('ij,kj->ki', rotation, points)
    mid = points[3].copy()
    mid[0] = 0
    rotation = __rotation(mid, np.array([0, la.norm(mid), 0]))
    points = np.einsum('ij,kj->ki', rotation, points)
    points = (points - np.mean(points, axis=0)).reshape((-1,))
    return points / la.norm(points)

def __diff(p,q):
    return [p[i] - q[i] for i in range(3)]

def __x(p,q):
    return [p[1]*q[2] - p[2]*q[1], p[2]*q[0] - p[0]*q[2], p[0]*q[1] - p[1]*q[0]]

def __dot(p,q):
    return p[0]*q[0] + p[1]*q[1] + p[2]*q[2]

def __n(v):
    tmp = sqrt(__dot(v,v))
    return [x/tmp for x in v]

def __rotation (v_from, v_to):
    v_from = v_from / la.norm(v_from)
    v_to = v_to / la.norm(v_to)
    prod = float(np.einsum('i,i->', v_from, v_to))
    v_aux = v_from - prod * v_to
    v_aux /= la.norm(v_aux)
    m1 = np.einsum('i,j->ij', v_aux, v_to)
    m1 = m1.T -m1
    m2 = np.einsum('i,j->ij', v_aux, v_aux) + np.einsum('i,j->ij', v_to, v_to)
    return np.eye(len(v_from)) + sqrt(1 - prod**2) * m1 + (prod -1) * m2

def import_m3d (filename):
    raw = None
    with open(filename) as datafile:
        raw = ''.join(''.join(line for line in datafile).split())
    iterator = iter(raw)
    def bracket_parser (level=0, d=None):
        word = ''
        if d is None:
            d = {}
        while True:
            try:
                char = next(iterator)
            except StopIteration:
                if level == 0:
                    return d
                raise Exception('missing closing bracket')
            if char == '}':
                if level > 0:
                    return d
                raise Exception('missing opening bracket')
            elif char == '{':
                d[word] = bracket_parser(level+1)
                word = ''
            elif char == ';':
                return word
            elif char == '=':
                d[word] = bracket_parser(level+1)
                word = ''
            else:
                word += char
    return bracket_parser()

def import_srep (filename):
    model = import_m3d(filename)['model']
    if model['figureCount'] != '1':
        raise Exception('Several figures in model')
    model = model['figure[0]']
    lattice = []
    spokes = []
    for node in sorted(model):
        m = model[node]
        if 'primitive' in node and m['selected'] == '1':
            lattice += [float(m['x']), float(m['y']), float(m['z'])]
            for i in range(3 if m['type'] == 'EndPrimitive' else 2):
                spokes.append(__spoke(m,i))
    return np.array(lattice), np.array(spokes)

def __spoke (m, i):
    return [float(m['r[%d]'%i]), float(m['ux[%d]'%i]),
            float(m['uy[%d]'%i]), float(m['uz[%d]'%i])]

def export_csv (data, filename, mode = 'Float'):
    form = '%.6f' if mode == 'Float' else ('%d' if mode == 'Int' else '%s')
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    with open(OUTPUT_FOLDER + filename, 'w') as output:
        for key in sorted(data.keys()):
            output.write('%%%s\n' % key + array2csv(data[key], form))
        output.close()

################################################################################
#########################   File and Path Operations   #########################
################################################################################

def find_files (pattern, folder='.'):
    filenames = []
    for root, dirs, files in os.walk(folder):
        for filename in fnmatch.filter(files, pattern):
            filenames.append(os.path.join(root, filename))
    return filenames

def sort_files (filenames):
    dictionary = {}
    for f in filenames:
        [folder, name] = os.path.split(f)
        folder = os.path.split(folder)[1]
        if not folder in dictionary:
            dictionary[folder] = []
        dictionary[folder].append(f)
    return dictionary

def crop_name (filename):
    return '.'.join(os.path.split(filename)[-1].split('.')[:-1])

def folder_name (invert, degenerate):
    folder = 'pns/'
    folder += ('inverse' if invert else 'standard')
    if not degenerate:
        folder += '_no_deg'
    return folder + '/'

################################################################################
###########################   Auxiliary Functions   ############################
################################################################################

def discard_empty_lists (some_list):
    some_list = [x for x in some_list if x]
    for sublist in some_list:
        if isinstance(sublist, list):
            discard_empty_lists (sublist)
    some_list = [x for x in some_list if x]
    return some_list

def array2csv (array, form = '%.6f', s=''):
    if len(array.shape) == 1:
        return s + ','.join(form % item for item in array) + '\n'
    if len(array.shape) == 2:
        return s + ('\n'+s).join(','.join(form % i for i in k) for k in array) + '\n'
    if len(array.shape) > 2:
        marker = ',' * (len(array.shape) - 1) + '\n'
        return s + marker.join([array2csv(m) for m in array])
    return None

def __get_all_pdbs ():
    files = find_files('*.pdb')
    data = np.vstack([x for x in [import_pdb_file(f) for f in files] if len(x.shape) == 2])
    print(data.shape)
    export_csv({'PDB-Data' : data}, '../data/PDB/RNA_data.csv')
    export_csv({'PDB-Data' : data[:,:7]}, '../data/PDB/RNA_data_7angles.csv')
    from RESH_plot import scatter_plots
    scatter_plots(data[:,:7],'test')
    scatter_plots(data[:,7:9],'test_')

def __get_all_pdbs_richardson ():
    files = find_files('*.pdb')
    data = np.vstack([x for x in [import_pdb_file_richardson(f) for f in files] if len(x.shape) == 2])
    print(data.shape)
    export_csv({'PDB-Data' : data}, '../data/PDB/RNA_data_richardson.csv')
    from RESH_plot import scatter_plots
    scatter_plots(data[:,:7],'test')

def __get_pdbs_extended (rna_set=None):
    all_files = find_files('*.pdb')
    files = []
    if rna_set is None:
        files = all_files
    else:
        for f in all_files:
            for name in rna_set:
                if name in f:
                    files.append(f)
                    break
    data = np.vstack([x for x in [import_pdb_file_extended(f) for f in files] if len(x.shape) == 3])
    data = np.array([data[:,:,0],data[:,:,1]])
    print(data.shape)
    name = ('tRNA' if (rna_set is T_RNAS) else
            ('tRNA_tt' if (rna_set is T_RNA_TT) else
             ('tRNA_ec' if (rna_set is T_RNA_EC) else
              ('ribosomes' if (rna_set is RIBOSOMES) else
               (rna_set[0] if len(rna_set) == 1 else 'unknown')))))
    export_csv({'PDB-Data' : data}, '../data/PDB/RNA_data_extended_' +
                                    name + '.csv')

def __get_all_S2 ():
    files = find_files('*.pdb')
    data = np.vstack([x for x in [import_pdb_file_to_S2(f) for f in files]
                      if len(x.shape) == 2])
    print(data.shape)
    export_csv({'PDB-Data' : data}, '../data/PDB/RNA_triangle_data.csv')

def __get_all_chains ():
    files = find_files('*.pdb')
    data = sum([import_chains_from_pdb_file(f) for f in files], [])
    from RESH_plot import scatter_plots
    for i,d in enumerate(data):
        if len(d) > 100:
            name = 'chain_%03d_length_%d' % (i, len(d))
            scatter_plots(d[:,:7], name + '_7d')
            scatter_plots(d[:,7:9], name + '_2d')
    data = {('chain%03d'%i):d for i,d in enumerate(data)}
    export_csv(data, '../data/PDB/RNA_chains_data.csv')

def __count_all ():
    files = find_files('*.pdb')
    for f in sorted(files):
        chains = import_chains_from_pdb_file(f)
        print([len(c) for c in chains], sum([len(c) for c in chains]))

def __get_tRNA_pdbs ():
    all_files = find_files('*.pdb')
    files = []
    for f in all_files:
        for name in T_RNAS:
            if name in f:
                files.append(f)
                break
    data = np.vstack([x for x in [import_pdb_file(f) for f in files] if len(x.shape) == 2])
    print(data.shape)
    export_csv({'PDB-Data' : data}, '../data/PDB/tRNA_data.csv')
    export_csv({'PDB-Data' : data[:,:7]}, '../data/PDB/tRNA_data_7angles.csv')
    from RESH_plot import scatter_plots
    scatter_plots(data[:,:7],'t-rna')
    scatter_plots(data[:,7:9],'t-rna_')

def __list_tRNA_pdbs ():
    all_files = find_files('*.pdb')
    files = []
    for f in all_files:
        for name in T_RNAS:
            if name in f:
                files.append(f)
                break
    data = {os.path.basename(f)[:4] : import_pdb_file(f) for f in files}
    export_csv(data, '../data/PDB/tRNA_listed_data2.csv')

def __get_all_pdb_hyperspheres ():
    files = find_files('*.pdb')
    data = np.vstack([x for x in [import_pdb_file_to_hypersphere(f) for f in files] if len(x.shape) == 2])
    print(data.shape)
    export_csv({'PDB-Data' : data}, '../data/PDB/RNA_hypersphere_data.csv')

if __name__ == '__main__':
#    table = import_lists(find_files('geoPCA*.txt')[0], 'String')
#    table_dict = {}
#    for [s,n,c] in table:
#        if not s.lower() in table_dict:
#            table_dict[s.lower()] = {}
#        if not c in table_dict[s.lower()]:
#            table_dict[s.lower()][c] = []
#        table_dict[s.lower()][c].append(int(n))
#    angle_dict = {}
#    for f in files:
#        for key in table_dict:
#            if key in f:
#                tmp_dict = import_from_pdb (f, table_dict[key], verbose=True)
#                for c in tmp_dict:
#                    if not c in angle_dict:
#                        angle_dict[c] = []
#                    angle_dict[c] += tmp_dict[c]
#                break
#    __get_all_pdbs()
#    __get_tRNA_pdbs()
#    __get_all_chains()
    #__count_all()
#    l, s = import_srep(find_files('*.m3d')[0])
#    print(l.shape, s.shape)
#    __get_pdbs_extended(T_RNA_TT)
#    __get_pdbs_extended(T_RNA_EC)
##    __get_all_S2()
#    __get_all_pdbs_richardson()
#    __list_tRNA_pdbs()
    __get_all_pdb_hyperspheres()
