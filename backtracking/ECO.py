
from collections import namedtuple, defaultdict, deque
from contextlib import contextmanager

from bits import *

Anchor = namedtuple('Anchor', ['symbol', 'stars'])
Star = namedtuple('Star', ['row', 'col', 'offsets', 'link', 'symbol'])
Structure = namedtuple('Structure', ['anchors', 'stars', 'shapes', 'tables'])

# numbering and coordinates are inverses the one of the other
def coordinates(group):

    def C(anchor):
        c, r = divmod(anchor, group)
        return r, c # simply flip them

    return C

def numbering(group):

    def N(coordinate):
        r, c = coordinate
        return r + group*c

    return N

#_______________________________________________________________

@contextmanager
def extract_star(stars, structure, locating):

    coordinates, numbering = locating

    anchor, stars = low_bit_and_clear(stars)
    r, c = coordinates(anchor)

    anchors_table, stars_table = structure.tables
    del stars_table[r, c]

    # get the corresponding `shape` object
    s = structure.shapes[r, c]
    del structure.shapes[r, c]

    # and "place" it, namely cell at position (r, c) will be occupied by a new anchor object.
    anchor_tuple = s(r, c)

    current_anchors_table, current_stars_table = dict(anchors_table), dict(stars_table)

    # remember the anchor symbol
    current_anchors_table[r, c] = anchor_tuple.symbol

    yield (anchor, stars), anchor_tuple, (current_anchors_table, current_stars_table)


#_______________________________________________________________

def recursive_structures(shapes_spec, start_cell, locating):
    "Returns a generator of structures according `shapes` recursion schemas, starting at anchor `init`."

    shapes_descr, initial_shape = shapes_spec
    coordinates, numbering = locating

    def gen(generation):

        yield generation

        next_generation = []

        for structure in generation:

            anchors, stars, shapes, (anchors_table, stars_table) = structure

            while stars: # `stars` is an integer to be used as bitmask: 1 in position k
                         # means that a ★ is present at cell numbered k

                with extract_star(stars, structure, locating) as (
                    (anchor, stars), anchor_tuple, (current_anchors_table, current_stars_table)):

                    # preparing for looping on stars and collecting new stars positions
                    augmented_stars, new_shapes = stars, dict(shapes)

                    for star in anchor_tuple.stars:
                        rs, cs, offsets = star.row, star.col, star.offsets

                        if (rs, cs) in anchors_table:
                            pass
                            #raise ValueError("Attempt to place a star in a cell already occupied by an anchor.")

                        if offsets:
                            r_offset, c_offset = offsets
                            translating_stars = ones_of(augmented_stars)
                            augmented_stars = 0
                            inner_stars_table = {}
                            inner_shapes = {}
                            for ts in translating_stars:
                                ts_r, ts_c = coordinates(ts)
                                symbol = current_stars_table[ts_r, ts_c]
                                shape = new_shapes[ts_r, ts_c]
                                translated = ts_r + r_offset, ts_c + c_offset
                                augmented_stars = set_bit(augmented_stars, numbering(translated))
                                inner_stars_table[translated] = symbol
                                inner_shapes[translated] = shape

                            current_stars_table = inner_stars_table
                            new_shapes = inner_shapes

                        current_stars_table[rs, cs] = star.symbol
                        new_shapes[rs, cs] = shapes_descr[star.link]
                        star_cell = numbering((rs,cs),)
                        augmented_stars = set_bit(augmented_stars, star_cell)


                    next_generation.append(Structure(anchors=set_bit(anchors, anchor),
                                                     stars=augmented_stars,
                                                     shapes=new_shapes,
                                                     tables=(current_anchors_table, current_stars_table)))

        yield from gen(next_generation)


    initial_structure = Structure(anchors=0b0,
                                  stars=1 << start_cell,
                                  shapes={coordinates(start_cell):shapes_descr[initial_shape]},
                                  tables=({}, {coordinates(start_cell):'★'}))

    return gen([initial_structure])

#________________________________________________________________________________________
def make_pretty(dim, joiner='', separator=',', empty=' '):

    def pretty(structures,):
        from collections import defaultdict

        rows, cols = dim

        strings = []
        for anchors, stars, shapes, (anchors_table, stars_table) in structures:

            table = defaultdict(lambda: empty)

            for k, v in anchors_table.items():
                table[k] = table[k] + separator + v if k in table else v

            for k, v in stars_table.items():
                table[k] = table[k] + separator + v if k in table else v

            s = ''
            for r in range(rows):
                s += joiner.join(table[r, c] for c in range(cols)) + '\n'

            strings.append(s)

        return  ''.join(strings)

    return pretty

# ______________________________________________________________________________

# In the following, the symbol ★ has a generation power,
# namely produces a new structure with the basic shape replacing ★ recursively.

"""
  ●
★   ★
"""
binary_tree_shapes = {
    'bintree': lambda r, c: Anchor(symbol='●', stars=[
        Star(row=r+1, col=c-1, offsets=None, link='bintree', symbol='★'),
        Star(row=r+1, col=c+1, offsets=None, link='bintree', symbol='★'),
    ]),
}

"""
  ●
★ ★ ★
"""
ternary_tree_shapes = {
    'threetree': lambda r, c: Anchor(symbol='●', stars=[
        Star(row=r+1, col=c-1, offsets=None, link='threetree', symbol='★'),
        Star(row=r+1, col=c, offsets=None, link='threetree', symbol='★'),
        Star(row=r+1, col=c+1, offsets=None, link='threetree', symbol='★'),
    ]),
}


"""
  ★
/   ★
"""
dyck_path_shapes = {
    'dick': lambda r, c: Anchor(symbol='/', stars=[
        Star(row=r-1, col=c+1, offsets=(0, 2), link='dick', symbol='★'),
        Star(row=r,   col=c+2, offsets=None,   link='dick', symbol='★'),
    ]),
}

"""
● ★ ★
"""
ballot_shapes = {
    'linear': lambda r, c: Anchor(symbol='●', stars=[
        Star(row=r, col=c+1, offsets=(0,2), link='linear', symbol='★'),
        Star(row=r, col=c+2, offsets=None,  link='linear', symbol='★'),
    ]),
}

"""
( ★ ) ★
"""
balanced_parens_shapes = {
    'parens': lambda r, c: Anchor(symbol='(', stars=[
        Star(row=r, col=c+1, offsets=(0, 3), link='parens', symbol='★'),
        Star(row=r, col=c+3, offsets=None,   link='parens', symbol='★'),
    ]),
}

"""
★
Λ
 ★
"""
plane_trees_shapes = {
    'plane': lambda r, c: Anchor(symbol='Λ', stars=[
        Star(row=r-1, col=c, offsets=None, link='plane', symbol='★'),
        Star(row=r+1, col=c+1, offsets=None, link='plane', symbol='★'),
    ]),
}


"""
n:    sw:    se:    nw:    ne:   s:
                     ★     ★
★▲★   ★▲     ▲★     ★▼     ▼★    ★▼★
       ★     ★
"""
triangulated_shapes = {
    'north': lambda r, c: Anchor(symbol='▲', stars=[
        Star(row=r, col=c-1, offsets=None, link='nw', symbol='★'),
        Star(row=r, col=c+1, offsets=None, link='ne', symbol='★'),
    ]),
    'sw':   lambda r, c: Anchor(symbol='▲', stars=[
        Star(row=r,   col=c-1, offsets=None, link='nw',    symbol='★'),
        Star(row=r+1, col=c,   offsets=None, link='south', symbol='★'),
    ]),
    'se':   lambda r, c: Anchor(symbol='▲', stars=[
        Star(row=r+1, col=c,   offsets=None, link='south', symbol='★'),
        Star(row=r,   col=c+1, offsets=None, link='ne',    symbol='★'),
    ]),
    'nw':   lambda r, c: Anchor(symbol='▼', stars=[
        Star(row=r,   col=c-1, offsets=None, link='sw',    symbol='★'),
        Star(row=r-1, col=c,   offsets=None, link='north', symbol='★'),
    ]),
    'ne':   lambda r, c: Anchor(symbol='▼', stars=[
        Star(row=r-1, col=c,   offsets=None, link='north', symbol='★'),
        Star(row=r,   col=c+1, offsets=None, link='se',    symbol='★'),
    ]),
    'south': lambda r, c: Anchor(symbol='▼', stars=[
        Star(row=r, col=c-1, offsets=None, link='sw', symbol='★'),
        Star(row=r, col=c+1, offsets=None, link='se', symbol='★'),
    ]),
}

"""
★
▢ ★
"""
blocks_shapes = {
    'block': lambda r, c: Anchor(symbol='▢', stars=[
        Star(row=r-1, col=c, offsets=None, link='block', symbol='★'),
        Star(row=r, col=c+1, offsets=None, link='block', symbol='★'),
    ]),
}


"""
  ★
○

● ▲
  ★
"""
rabbits_shapes = {
    'young': lambda r, c: Anchor(symbol='○', stars=[
        Star(row=r-1, col=c+1, offsets=None, link='senior', symbol='★'),
    ]),
    'senior': lambda r, c: Anchor(symbol='●', stars=[
        Star(row=r,   col=c+1, offsets=None, link='young',  symbol='▲'),
        Star(row=r+1, col=c+1, offsets=None, link='senior', symbol='★'),
    ]),
}


"""
☆
▢

☆
▢ ★
"""
steep_shapes = {
    'one_star': lambda r, c: Anchor(symbol='▢', stars=[
        Star(row=r-1, col=c, offsets=None, link='two_stars', symbol='☆'),
    ]),
    'two_stars': lambda r, c: Anchor(symbol='▢', stars=[
        Star(row=r-1, col=c,   offsets=None, link='two_stars', symbol='☆'),
        Star(row=r,   col=c+1, offsets=None, link='one_star',  symbol='★'),
    ]),
}

"""
  ★
― o o
    ★
"""
sigmoid_shapes = {
    'sigmoid': lambda r, c: Anchor(symbol='―', stars=[
        Star(row=r-1, col=c+1, offsets=None, link='sigmoid', symbol='★'),
        Star(row=r+1, col=c+2, offsets=None, link='sigmoid', symbol='★'),
    ]),
}


"""
★
─  :horizontal
★

★ │ ★   :vertical
"""
splitters_shapes = {
    'horizontal': lambda r, c: Anchor(symbol='─', stars=[
        Star(row=r-1, col=c, offsets=None, link='vertical', symbol='★'),
        Star(row=r+1, col=c, offsets=None, link='vertical', symbol='★'),
    ]),
    'vertical': lambda r, c: Anchor(symbol='│', stars=[
        Star(row=r, col=c-1, offsets=None, link='horizontal', symbol='★'),
        Star(row=r, col=c+1, offsets=None, link='horizontal', symbol='★'),
    ]),
}

# ______________________________________________________________________________


def doctests():
    """
    Doctests, simply.

    >>> rows, cols = 8, 25
    >>> coor, number = coordinates(rows), numbering(rows)

    Binary trees
    ============

    Starts at the first cell of the 13th column
    >>> bin_trees = recursive_structures((binary_tree_shapes, 'bintree'), rows*12, (coor, number))
    >>> representations = map(make_pretty((rows, cols),), bin_trees)
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
                ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
                ●
               ★ ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
                ●
               ● ★
              ★ ★
    <BLANKLINE>
                ●
                 ●
                ★ ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
                ●
               ● ★
              ● ★
             ★ ★
    <BLANKLINE>
                ●
               ● ★
                ●
               ★ ★
    <BLANKLINE>
                ●
               ● ●
                ★ ★
    <BLANKLINE>
                ●
                 ●
                ● ★
               ★ ★
    <BLANKLINE>
                ●
                 ●
                  ●
                 ★ ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
                ●
               ● ★
              ● ★
             ● ★
            ★ ★
    <BLANKLINE>
                ●
               ● ★
              ● ★
               ●
              ★ ★
    <BLANKLINE>
                ●
               ● ★
              ● ●
               ★ ★
    <BLANKLINE>
                ●
               ● ●
              ● ★ ★
    <BLANKLINE>
                ●
               ● ★
                ●
               ● ★
              ★ ★
    <BLANKLINE>
                ●
               ● ●
                ●,★ ★
                 ★
    <BLANKLINE>
                ●
               ●
                ●
                 ●
                ★ ★
    <BLANKLINE>
                ●
               ● ●
                ● ★
               ★ ★
    <BLANKLINE>
                ●
               ● ●
                  ●
                 ★ ★
    <BLANKLINE>
                ●
                 ●
                ● ★
               ● ★
              ★ ★
    <BLANKLINE>
                ●
                 ●
                ● ★
                 ●
                ★ ★
    <BLANKLINE>
                ●
                 ●
                ● ●
                 ★ ★
    <BLANKLINE>
                ●
                 ●
                  ●
                 ● ★
                ★ ★
    <BLANKLINE>
                ●
                 ●
                  ●
                   ●
                  ★ ★


    Ternary trees
    =============

    Starts at the first cell of the 13th column:
    >>> three_trees = recursive_structures((ternary_tree_shapes, 'threetree'), rows*12, (coor, number))
    >>> representations = map(make_pretty((rows, cols),), three_trees)
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
                ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
                ●
               ★★★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
                ●
               ●★★
              ★★★
    <BLANKLINE>
                ●
                ●★
               ★★★
    <BLANKLINE>
                ●
                 ●
                ★★★
    <BLANKLINE>
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
                ●
               ●★★
              ●★★
             ★★★
    <BLANKLINE>
                ●
               ●★★
               ●★
              ★★★
    <BLANKLINE>
                ●
               ●●★
               ★★★
    <BLANKLINE>
                ●
               ● ★
                ●
               ★★★
    <BLANKLINE>
                ●
               ● ●
                ★★★
    <BLANKLINE>
                ●
                ●★
               ●★★
              ★★★
    <BLANKLINE>
                ●
                ●★
                ●★
               ★★★
    <BLANKLINE>
                ●
                ●●
                ★★★
    <BLANKLINE>
                ●
                ●
                 ●
                ★★★
    <BLANKLINE>
                ●
                 ●
                ●★★
               ★★★
    <BLANKLINE>
    <BLANKLINE>
    <BLANKLINE>
    <BLANKLINE>
                ●
                 ●
                 ●★
                ★★★
    <BLANKLINE>
                ●
                 ●
                  ●
                 ★★★
    <BLANKLINE>

    Dyck Paths
    ==========

    >>> dick_paths = recursive_structures((dyck_path_shapes, 'dick'), rows-1, (coor, number))
    >>> representations = map(make_pretty((rows, cols),), dick_paths)
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
    ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
     ★
    / ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
      ★
     / ★
    /   ★
    <BLANKLINE>
       ★
    / / ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
       ★
      / ★
     /   ★
    /     ★
    <BLANKLINE>
        ★
     / / ★
    /     ★
    <BLANKLINE>
     /   ★
    /   / ★
    <BLANKLINE>
        ★
       / ★
    / /   ★
    <BLANKLINE>
         ★
    / / / ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
        ★
       / ★
      /   ★
     /     ★
    /       ★
    <BLANKLINE>
         ★
      / / ★
     /     ★
    /       ★
    <BLANKLINE>
      /   ★
     /   / ★
    /       ★
    <BLANKLINE>
      /
     /     ★
    /     / ★
    <BLANKLINE>
         ★
        / ★
     / /   ★
    /       ★
    <BLANKLINE>
          ★
     / / / ★
    /       ★
    <BLANKLINE>
     / /   ★
    /     / ★
    <BLANKLINE>
          ★
     /   / ★
    /   /   ★
    <BLANKLINE>
     /     ★
    /   / / ★
    <BLANKLINE>
         ★
        / ★
       /   ★
    / /     ★
    <BLANKLINE>
          ★
       / / ★
    / /     ★
    <BLANKLINE>
       /   ★
    / /   / ★
    <BLANKLINE>
          ★
         / ★
    / / /   ★
    <BLANKLINE>
           ★
    / / / / ★

    Ballots
    =======

    >>> rows, cols = 3, 25
    >>> coor, number = coordinates(rows), numbering(rows)

    >>> linears = recursive_structures((ballot_shapes, 'linear'), 1, (coor, number))
    >>> representations = map(make_pretty((rows, cols),), linears)
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
    ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
    ●★★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
    ●●★★★
    <BLANKLINE>
    ● ●★★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
    ●●●★★★★
    <BLANKLINE>
    ●● ●★★★
    <BLANKLINE>
    ●●  ●★★
    <BLANKLINE>
    ● ●●★★★
    <BLANKLINE>
    ● ● ●★★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
    ●●●●★★★★★
    <BLANKLINE>
    ●●● ●★★★★
    <BLANKLINE>
    ●●●  ●★★★
    <BLANKLINE>
    ●●●   ●★★
    <BLANKLINE>
    ●● ●●★★★★
    <BLANKLINE>
    ●● ● ●★★★
    <BLANKLINE>
    ●● ●  ●★★
    <BLANKLINE>
    ●●  ●●★★★
    <BLANKLINE>
    ●●  ● ●★★
    <BLANKLINE>
    ● ●●●★★★★
    <BLANKLINE>
    ● ●● ●★★★
    <BLANKLINE>
    ● ●●  ●★★
    <BLANKLINE>
    ● ● ●●★★★
    <BLANKLINE>
    ● ● ● ●★★

    Balanced parens
    ===============

    >>> rows, cols = 3, 32
    >>> coor, number = coordinates(rows), numbering(rows)

    >>> parens = recursive_structures((balanced_parens_shapes, 'parens'), 1, (coor, number))
    >>> representations = map(make_pretty((rows, cols),), parens)
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
    ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
    (★ ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
    ((★ ★ ★
    <BLANKLINE>
    (  (★ ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
    (((★ ★ ★ ★
    <BLANKLINE>
    ((  (★ ★ ★
    <BLANKLINE>
    ((    (★ ★
    <BLANKLINE>
    (  ((★ ★ ★
    <BLANKLINE>
    (  (  (★ ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
    ((((★ ★ ★ ★ ★
    <BLANKLINE>
    (((  (★ ★ ★ ★
    <BLANKLINE>
    (((    (★ ★ ★
    <BLANKLINE>
    (((      (★ ★
    <BLANKLINE>
    ((  ((★ ★ ★ ★
    <BLANKLINE>
    ((  (  (★ ★ ★
    <BLANKLINE>
    ((  (    (★ ★
    <BLANKLINE>
    ((    ((★ ★ ★
    <BLANKLINE>
    ((    (  (★ ★
    <BLANKLINE>
    (  (((★ ★ ★ ★
    <BLANKLINE>
    (  ((  (★ ★ ★
    <BLANKLINE>
    (  ((    (★ ★
    <BLANKLINE>
    (  (  ((★ ★ ★
    <BLANKLINE>
    (  (  (  (★ ★

    Plane trees
    ===========

    >>> rows, cols = 10, 25
    >>> coor, number = coordinates(rows), numbering(rows)

    >>> plane_trees = recursive_structures((plane_trees_shapes, 'plane'), 4, (coor, number))
    >>> representations = map(make_pretty((rows, cols),), plane_trees)
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
    ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
    ★
    Λ
     ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
    ★
    Λ
    Λ★
     ★
    <BLANKLINE>
    Λ★
     Λ
      ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
    ★
    Λ
    Λ★
    Λ★
     ★
    <BLANKLINE>
    Λ★
    ΛΛ
     ★★
    <BLANKLINE>
    Λ
    Λ★
     Λ
      ★
    <BLANKLINE>
     ★
    ΛΛ
     Λ★
      ★
    <BLANKLINE>
    Λ
     Λ★
      Λ
       ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
    ★
    Λ
    Λ★
    Λ★
    Λ★
     ★
    <BLANKLINE>
    Λ★
    ΛΛ
    Λ★★
     ★
    <BLANKLINE>
    Λ
    Λ★
    ΛΛ
     ★★
    <BLANKLINE>
    Λ
    Λ
    Λ★
     Λ
      ★
    <BLANKLINE>
     ★
    ΛΛ
    ΛΛ★
     ★★
    <BLANKLINE>
    Λ
    ΛΛ,★
     Λ★
      ★
    <BLANKLINE>
    Λ
    ΛΛ★
      Λ
       ★
    <BLANKLINE>
    Λ★
    ΛΛ
     Λ★
      ★
    <BLANKLINE>
    Λ
    Λ
     Λ★
      Λ
       ★
    <BLANKLINE>
     ★
     Λ
    ΛΛ★
     Λ★
      ★
    <BLANKLINE>
    ΛΛ★
     ΛΛ
      ★★
    <BLANKLINE>
    ΛΛ
     Λ★
      Λ
       ★
    <BLANKLINE>
    Λ ★
     ΛΛ
      Λ★
       ★
    <BLANKLINE>
    Λ
     Λ
      Λ★
       Λ
        ★

    Triangulated polygons
    =====================

    >>> rows, cols = 8, 25
    >>> coor, number = coordinates(rows), numbering(rows)

    >>> triangulated = recursive_structures((triangulated_shapes, 'north'), rows*12+4, (coor, number))
    >>> representations = map(make_pretty((rows, cols),), triangulated)
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
                ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
               ★▲★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
               ★
              ★▼▲★
    <BLANKLINE>
                 ★
                ▲▼★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
               ★
             ★▲▼▲★
              ★
    <BLANKLINE>
              ★▲★
               ▼▲★
    <BLANKLINE>
                 ★
               ▼▲▼★
    <BLANKLINE>
                ★▲★
                ▲▼★
    <BLANKLINE>
                ▲▼▲★
                  ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
             ★ ★
            ★▼▲▼▲★
              ★
    <BLANKLINE>
               ★
              ▲▼▲★
             ★▼★
    <BLANKLINE>
              ★▲★
              ▲▼▲★
    <BLANKLINE>
                 ★
              ▲▼▲▼★
    <BLANKLINE>
              ★
             ★▼▲★
               ▼▲★
    <BLANKLINE>
                ★
               ▲▼★
               ▼▲★
    <BLANKLINE>
               ▲ ★
               ▼▲▼★
    <BLANKLINE>
                ★▲★
               ▼▲▼★
    <BLANKLINE>
               ▼▲▼▲★
                  ★
    <BLANKLINE>
                ★
               ★▼▲★
                ▲▼★
    <BLANKLINE>
                  ★
                 ▲▼★
                ▲▼★
    <BLANKLINE>
                 ▲
                ▲▼▲★
                  ★
    <BLANKLINE>
                ▲▼▲★
                 ★▼★
    <BLANKLINE>
                   ★
                ▲▼▲▼★


    Blocks
    ======

    >>> rows, cols = 10, 25
    >>> coor, number = coordinates(rows), numbering(rows)

    >>> blocks = recursive_structures((blocks_shapes, 'block'), 4, (coor, number))
    >>> representations = map(make_pretty((rows, cols), joiner=' '), blocks)
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
    ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
    ★
    ▢ ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
    ★
    ▢ ★
    ▢ ★
    <BLANKLINE>
      ★
    ▢ ▢ ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
    ★
    ▢ ★
    ▢ ★
    ▢ ★
    <BLANKLINE>
      ★
    ▢ ▢ ★
    ▢ ★
    <BLANKLINE>
    ▢ ★
    ▢ ▢ ★
    <BLANKLINE>
      ★
      ▢ ★
    ▢ ▢ ★
    <BLANKLINE>
        ★
    ▢ ▢ ▢ ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
    ★
    ▢ ★
    ▢ ★
    ▢ ★
    ▢ ★
    <BLANKLINE>
      ★
    ▢ ▢ ★
    ▢ ★
    ▢ ★
    <BLANKLINE>
    ▢ ★
    ▢ ▢ ★
    ▢ ★
    <BLANKLINE>
    ▢
    ▢ ★
    ▢ ▢ ★
    <BLANKLINE>
      ★
      ▢ ★
    ▢ ▢ ★
    ▢ ★
    <BLANKLINE>
    ▢ ▢,★ ★
    ▢ ▢ ★
    <BLANKLINE>
        ★
    ▢ ▢ ▢ ★
    ▢
    <BLANKLINE>
      ★
    ▢ ▢ ★
    ▢ ▢ ★
    <BLANKLINE>
    ▢   ★
    ▢ ▢ ▢ ★
    <BLANKLINE>
      ★
      ▢ ★
      ▢ ★
    ▢ ▢ ★
    <BLANKLINE>
        ★
      ▢ ▢ ★
    ▢ ▢ ★
    <BLANKLINE>
      ▢ ★
    ▢ ▢ ▢ ★
    <BLANKLINE>
        ★
        ▢ ★
    ▢ ▢ ▢ ★
    <BLANKLINE>
          ★
    ▢ ▢ ▢ ▢ ★


    Rabbits
    =======

    >>> rows, cols = 8, 25
    >>> coor, number = coordinates(rows), numbering(rows)

    >>> rabbits = recursive_structures((rabbits_shapes, 'young'), 4, (coor, number))
    >>> representations = map(make_pretty((rows, cols), joiner=''), rabbits)

    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
    ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
     ★
    ○
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
     ●▲
    ○ ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
       ★
     ●○
    ○ ★
    <BLANKLINE>
     ●
    ○ ●▲
       ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
       ★
     ●○
    ○ ●▲
       ★
    <BLANKLINE>
       ●▲
     ●○ ★
    ○
    <BLANKLINE>
     ●  ★
    ○ ●○
       ★
    <BLANKLINE>
     ●
    ○ ●
       ●▲
        ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
       ●▲
     ●○ ★
    ○ ●▲
       ★
    <BLANKLINE>
     ●○ ★
    ○ ●○
       ★
    <BLANKLINE>
     ●○
    ○ ●
       ●▲
        ★
    <BLANKLINE>
         ★
       ●○
     ●○ ★
    ○
    <BLANKLINE>
       ●
     ●○ ●▲
    ○    ★
    <BLANKLINE>
     ●  ★
    ○ ●○
       ●▲
        ★
    <BLANKLINE>
     ●  ●▲
    ○ ●○ ★
    <BLANKLINE>
     ●
    ○ ●  ★
       ●○
        ★
    <BLANKLINE>
     ●
    ○ ●
       ●
        ●▲
         ★

    Steep parallelograms
    ====================

    >>> rows, cols = 14, 25
    >>> coor, number = coordinates(rows), numbering(rows)

    >>> steep_parallelograms = recursive_structures((steep_shapes, 'one_star'), 7, (coor, number))
    >>> representations = map(make_pretty((rows, cols), joiner=' '), steep_parallelograms)
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
    ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
    ☆
    ▢
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
    ☆
    ▢ ★
    ▢
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
    ☆
    ▢ ★
    ▢ ★
    ▢
    <BLANKLINE>
      ☆
    ▢ ▢
    ▢
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
    ☆
    ▢ ★
    ▢ ★
    ▢ ★
    ▢
    <BLANKLINE>
      ☆
    ▢ ▢
    ▢ ★
    ▢
    <BLANKLINE>
    ▢ ☆
    ▢ ▢
    ▢
    <BLANKLINE>
      ☆
      ▢ ★
    ▢ ▢
    ▢
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
    ☆
    ▢ ★
    ▢ ★
    ▢ ★
    ▢ ★
    ▢
    <BLANKLINE>
      ☆
    ▢ ▢
    ▢ ★
    ▢ ★
    ▢
    <BLANKLINE>
    ▢ ☆
    ▢ ▢
    ▢ ★
    ▢
    <BLANKLINE>
    ▢
    ▢ ☆
    ▢ ▢
    ▢
    <BLANKLINE>
      ☆
      ▢ ★
    ▢ ▢
    ▢ ★
    ▢
    <BLANKLINE>
    ▢ ▢,☆
    ▢ ▢
    ▢
    <BLANKLINE>
      ☆
    ▢ ▢ ★
    ▢ ▢
    ▢
    <BLANKLINE>
      ☆
      ▢ ★
      ▢ ★
    ▢ ▢
    ▢
    <BLANKLINE>
        ☆
      ▢ ▢
    ▢ ▢
    ▢

    Sigmoids
    ========

    >>> rows, cols = 9, 25
    >>> coor, number = coordinates(rows), numbering(rows)

    >>> sigmoids = recursive_structures((sigmoid_shapes, 'sigmoid'), 4, (coor, number))
    >>> representations = map(make_pretty((rows, cols),), sigmoids)
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
    ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
     ★
    ―
      ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
      ★
     ―
    ―  ★
      ★
    <BLANKLINE>
    ―  ★
      ―
        ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
       ★
      ―
     ―  ★
    ―  ★
      ★
    <BLANKLINE>
     ―
    ―  ★
      ―
        ★
    <BLANKLINE>
     ―  ★
    ―  ―
         ★
    <BLANKLINE>
        ★
    ―  ―
      ―  ★
        ★
    <BLANKLINE>
    ―
      ―  ★
        ―
          ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
       ★
      ―
     ―  ★
    ―  ★
      ―
        ★
    <BLANKLINE>
        ★
       ―
      ―  ★
     ―  ★
    ―  ★
    <BLANKLINE>
      ―
     ―  ★
    ―  ―
         ★
    <BLANKLINE>
      ―  ★
     ―  ―
    ―     ★
    <BLANKLINE>
     ―  ★
    ―  ―
      ―  ★
        ★
    <BLANKLINE>
     ―
    ―
      ―  ★
        ―
          ★
    <BLANKLINE>
         ★
     ―  ―
    ―  ―  ★
         ★
    <BLANKLINE>
     ―
    ―  ―  ★
         ―
           ★
    <BLANKLINE>
         ★
        ―
    ―  ―  ★
      ―  ★
        ★
    <BLANKLINE>
    ―  ―
      ―  ★
        ―
          ★
    <BLANKLINE>
    ―  ―  ★
      ―  ―
           ★
    <BLANKLINE>
    ―     ★
      ―  ―
        ―  ★
          ★
    <BLANKLINE>
    ―
      ―
        ―  ★
          ―
            ★

    Splitters
    =========

    >>> rows, cols = 8, 25
    >>> coor, number = coordinates(rows), numbering(rows)

    >>> splitters = recursive_structures((splitters_shapes, 'vertical'), rows*12+4, (coor, number))
    >>> representations = map(make_pretty((rows, cols),), splitters)
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
                ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
               ★│★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
               ★
               ─│★
               ★
    <BLANKLINE>
                 ★
                │─
                 ★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
              ★│★
               ─│★
               ★
    <BLANKLINE>
               ─│★
              ★│★
    <BLANKLINE>
                 ★
               ─│─
                 ★
    <BLANKLINE>
                ★│★
                │─
                 ★
    <BLANKLINE>
                │─
                ★│★
    >>> print(next(representations)) # doctest: +NORMALIZE_WHITESPACE
              ★
              ─│★
              ★─│★
               ★
    <BLANKLINE>
               │★
               ─│★
              ★│★
    <BLANKLINE>
                ★
               │─
               ─│,★★
    <BLANKLINE>
               │ ★
               ─│─
                 ★
    <BLANKLINE>
              ★─│★
              ─│★
              ★
    <BLANKLINE>
               ─│,★★
               │─
                ★
    <BLANKLINE>
                 ★
               ─│─
               │ ★
    <BLANKLINE>
                ★│★
               ─│─
                 ★
    <BLANKLINE>
               ─│─
                ★│★
    <BLANKLINE>
                ★
                ─│★
                │,★─
                 ★
    <BLANKLINE>
                 │★
                │─
                ★│★
    <BLANKLINE>
                  ★
                 │─
                │─★
    <BLANKLINE>
                │,★─
                ─│★
                ★
    <BLANKLINE>
                │─★
                 │─
                  ★

    """
    pass



