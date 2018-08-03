
import math, random, functools
from bits import *
from collections import namedtuple

shape_spec = namedtuple('shape_spec', ['name', 'isomorphisms',])

def polyominoes(dim, shapes, availables='ones',
                max_depth_reached=None, forbidden=[],
                pruning=lambda coord, positions, shapes: False):

    rows, cols = dim
    sol = []

    if not availables or availables == 'ones':
        availables = {s.name:1 for s in shapes}
    elif availables == 'inf':
        availables = {s.name:-1 for s in shapes}

    def place(S, positions):
        for r, c in positions:
            S = clear_bit(S, r + rows*c)
        return S

    def shapes_available():
        return {s for s in shapes if availables[s.name]}

    def gen(positions, attempts):

        p = low_bit(positions)
        c, r = divmod(p, rows)

        if pruning((r,c), positions, shapes_available()):
            raise StopIteration()

        for i, s in enumerate(shapes):

            if not availables[s.name]: continue

            for j, iso in enumerate(s.isomorphisms(r, c)):

                if all(0 <= rr < rows
                       and 0 <= cc < cols
                       and is_on(positions, rr + rows*cc)
                       for rr, cc in iso):

                    fewer_positions = place(positions, iso)
                    availables[s.name] -= 1
                    sol.append((s, positions, (r,c), iso),)

                    if not (fewer_positions and attempts):
                        yield sol
                    else:
                        yield from gen(fewer_positions, attempts-1)

                    sol.pop()
                    availables[s.name] += 1

    return gen(place(set_all(rows*cols), forbidden),
               max_depth_reached or -1)

def symbols_pretty( sol, dim, symbols, missing_symbol=' ',
                    joiner=lambda board: '\n'.join(board), axis=False):

    from collections import defaultdict

    table = defaultdict(lambda: missing_symbol)
    for s, _, _, iso in sol:
        table.update({(r, c): symbols[s.name] for r, c in iso})

    rows, cols = dim
    board = []

    board.append("┌" + ("─" * (2*cols + 1)) + (">" if axis else "┐"))
    for r in range(rows):
        board.append('│ ' + ' '.join(table[r, c] for c in range(cols))
                     + ('  ' if axis else ' │'))
    board.append("v" + (" " * 2*(cols+1)) if axis
                 else "└" + ("─" * (2*cols + 1)) + "┘")

    return joiner(board)


def tiling_pretty(  sol, dim, symbols, missing_symbol='●',
                    joiner=lambda board: '\n'.join(board)):

    from string import whitespace

    table = {}
    for s, _, _, iso in sol:
        table.update({(r,c):s.name for r, c in iso})

    rows, cols = dim

    matrix = []
    matrix.append(' ' + ' '.join(['_' if (0, c) in table else ' '
                                  for c in range(cols)]))

    for r in range(rows):
        row = '|'
        for c in range(cols):

            if (r, c) not in table:
                row += (missing_symbol +
                        ('|' if (r, c+1) in table else ' '))
            else:
                if (r+1, c) in table:
                    row += '_' if table[r, c] != table[r+1, c] else ' '
                else:
                    row += '_'

                if (r, c+1) in table:
                    row += '|' if table[r, c] != table[r, c+1] else ' '
                else:
                    row += '|'

        matrix.append(row)

    return joiner(matrix)

def lower_greek_symbols(plain_string=True):
    """
    Returns an iterable of *lower* Greek symbols.

    Built by Vim's `:help digraph-table@en` command.
    """
    return 'αβγδεζηθικλμνξοπρςστυφχψω'

def capital_greek_symbols():
    """
    Returns an iterable of *capital* Greek symbols.

    Built by Vim's `:help digraph-table@en` command.
    """
    return "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"

def light_box_drawings():
    """
    Return an iterable of *light* box symbols.

    Built by Vim's `:help digraph-table@en` command.
    """
    return "─│├┤┬┴┼┌┐└┘"


def markdown_pretty(tilings, dim, shapes, raw_text=False,
                    pretty_printer=tiling_pretty,
                    symbols=(lower_greek_symbols()
                             + capital_greek_symbols())):

    from IPython.display import Markdown
    import itertools

    def row_tabulator(row):
        # '\t' necessary for nice Markdown rendering
        return ('' if raw_text else '\t') + row

    for i in itertools.count():

        # since I'm a generator on top of `tiling`, which is a generator too,
        # if it raises `StopIteration`, I propagate that as well.
        #with timing(lambda: next(tilings)) as (s, elapsed_time):
            s = next(tilings)
            prettier = pretty_printer(s, dim, { s.name:symbols[i]
                                                for i,s in enumerate(shapes)},
                                      joiner=lambda board: '\n'.join(map(row_tabulator, board)))

            #code = "Solution $s_{{{}}}$ computed in __{}__ time, respect last solution elapsed time (which is 0 if the current is the very first one):\n{}".format(i, elapsed_time, prettier)

            yield prettier if raw_text else Markdown(code)

"""
  *
* * *
  *
"""
X_shape = shape_spec(name='X',
                     isomorphisms=lambda r, c: [((r, c), (r-1, c+1), (r, c+1), (r+1, c+1), (r, c+2))])

"""
*
*
*
*
*

* * * * *
"""
I_shape = shape_spec(name='I',
                     isomorphisms=lambda r, c: [((r, c), (r+1,c), (r+2,c), (r+3,c), (r+4,c)),
                                                ((r, c), (r,c+1), (r,c+2), (r,c+3), (r,c+4))])

"""
*      * * *       *   * * *
*          *       *   *
* * *      *   * * *   *
"""
V_shape = shape_spec(
            name='V',
            isomorphisms=lambda r, c: [
                ((r,c), (r+1,c),  (r+2,c),   (r+2, c+1), (r+2, c+2)),
                ((r,c), (r, c+1), (r,c+2),   (r+1, c+2), (r+2, c+2)),
                ((r,c), (r,c+1),  (r-2,c+2), (r-1,c+2),  (r, c+2)),
                ((r,c), (r+1,c),  (r+2,c),   (r,c+1),    (r, c+2))
            ])

"""
* *
*
* *

* * *
*   *

*   *
* * *

* *
  *
* *
"""
U_shape = shape_spec(name='U',
                     isomorphisms=lambda r, c: [((r,c), (r+1,c), (r+2,c), (r,c+1), (r+2,c+1)),
                                                ((r,c), (r+1,c), (r, c+1), (r,c+2), (r+1,c+2)),
                                                ((r,c), (r+1, c), (r+1, c+1), (r, c+2), (r+1, c+2)),
                                                ((r,c), (r+2, c), (r, c+1), (r+1, c+1), (r+2, c+1)),])

"""
*
* *
  * *

* *
  * *
    *

    *
  * *
* *

  * *
* *
*
"""
W_shape = shape_spec(name='W',
                     isomorphisms=lambda r, c: [((r,c), (r+1,c), (r+1,c+1), (r+2,c+1), (r+2,c+2)),
                                                ((r,c), (r,c+1), (r+1,c+1), (r+1,c+2), (r+2,c+2)),
                                                ((r,c), (r-1,c+1), (r,c+1), (r-2,c+2), (r-1,c+2)),
                                                ((r,c), (r+1,c), (r-1,c+1), (r,c+1), (r-1,c+2)),])

"""
* * *
  *
  *

*
* * *
*

    *
* * *
    *

  *
  *
* * *
"""
T_shape = shape_spec(name='T',
                     isomorphisms=lambda r, c: [((r,c), (r,c+1), (r+1,c+1), (r+2,c+1), (r,c+2)),
                                                ((r,c), (r+1,c), (r+2,c), (r+1,c+1), (r+1,c+2)),
                                                ((r,c), (r,c+1), (r-1,c+2), (r,c+2), (r+1,c+2)),
                                                ((r,c), (r-2,c+1), (r-1,c+1), (r,c+1), (r,c+2)),])

"""
*
* * *
    *

    *
* * *
*

  * *
  *
* *

* *
  *
  * *
"""
Z_shape = shape_spec(name='Z',
                     isomorphisms=lambda r, c: [((r,c), (r+1,c), (r+1,c+1), (r+1,c+2), (r+2,c+2)),
                                                ((r,c), (r+1,c), (r,c+1), (r-1,c+2), (r,c+2)),
                                                ((r,c), (r-2,c+1), (r-1,c+1), (r,c+1), (r-2,c+2)),
                                                ((r,c), (r,c+1), (r+1,c+1), (r+2,c+1), (r+2,c+2)),])

"""
*
* *
  *
  *

  *
* *
*
*

*
*
* *
  *

  *
  *
* *
*

  * * *
* *

* *
  * * *

* * *
    * *

    * *
* * *
"""
N_shape = shape_spec(name='N',
                     isomorphisms=lambda r, c: [((r,c), (r+1,c), (r+1,c+1), (r+2,c+1), (r+3,c+1)),
                                                ((r,c), (r+1,c), (r+2,c), (r-1,c+1), (r,c+1)),
                                                ((r,c), (r+1,c), (r+2,c), (r+2,c+1), (r+3,c+1)),
                                                ((r,c), (r+1,c), (r-2,c+1), (r-1,c+1), (r,c+1)),
                                                ((r,c), (r-1,c+1), (r,c+1), (r-1,c+2), (r-1,c+3)),
                                                ((r,c), (r,c+1), (r+1,c+1), (r+1,c+2), (r+1,c+3)),
                                                ((r,c), (r,c+1), (r,c+2), (r+1,c+2), (r+1,c+3)),
                                                ((r,c), (r,c+1), (r-1,c+2), (r,c+2), (r-1,c+3)),])

"""
*
*
*
* *

  *
  *
  *
* *

* *
  *
  *
  *

* *
*
*
*

*
* * * *

* * * *
*

      *
* * * *

* * * *
      *

"""
L_shape = shape_spec(name='L',
                     isomorphisms=lambda r, c: [((r,c), (r+1,c), (r+2,c), (r+3,c), (r+3,c+1)),
                                                ((r,c), (r-3,c+1), (r-2,c+1), (r-1,c+1), (r,c+1)),
                                                ((r,c), (r,c+1), (r+1,c+1), (r+2,c+1), (r+3,c+1)),
                                                ((r,c), (r+1,c), (r+2,c), (r+3,c), (r,c+1)),
                                                ((r,c), (r+1,c), (r+1,c+1), (r+1,c+2), (r+1,c+3)),
                                                ((r,c), (r+1,c), (r,c+1), (r,c+2), (r,c+3)),
                                                ((r,c), (r,c+1), (r,c+2), (r-1,c+3), (r,c+3)),
                                                ((r,c), (r,c+1), (r,c+2), (r,c+3), (r+1,c+3)),])

"""
*
* *
*
*

*
*
* *
*

  *
* *
  *
  *

  *
  *
* *
  *

* * * *
    *

    *
* * * *

* * * *
  *

  *
* * * *
"""
Y_shape = shape_spec(name='Y',
                     isomorphisms=lambda r, c: [((r,c), (r+1,c), (r+2,c), (r+3,c), (r+1,c+1)),
                                                ((r,c), (r+1,c), (r+2,c), (r+3,c), (r+2,c+1)),
                                                ((r,c), (r-1,c+1), (r,c+1), (r+1,c+1), (r+2,c+1)),
                                                ((r,c), (r-2,c+1), (r-1,c+1), (r,c+1), (r+1,c+1)),
                                                ((r,c), (r,c+1), (r,c+2), (r+1,c+2), (r,c+3)),
                                                ((r,c), (r,c+1), (r-1,c+2), (r,c+2), (r,c+3)),
                                                ((r,c), (r,c+1), (r+1,c+1), (r,c+2), (r,c+3)),
                                                ((r,c), (r-1,c+1), (r,c+1), (r,c+2), (r,c+3)),])

"""
*
* * *
  *

    *
* * *
  *

  *
* * *
*

  *
* * *
    *

  *
  * *
* *

  *
* *
  * *

* *
  * *
  *

  * *
* *
  *
"""
F_shape = shape_spec(name='F',
                     isomorphisms=lambda r, c: [((r,c), (r+1,c), (r+1,c+1), (r+2,c+1), (r+1,c+2)),
                                                ((r,c), (r,c+1), (r+1,c+1), (r-1,c+2), (r,c+2)),
                                                ((r,c), (r+1,c), (r-1,c+1), (r,c+1), (r,c+2)),
                                                ((r,c), (r-1,c+1), (r,c+1), (r,c+2), (r+1,c+2)),
                                                ((r,c), (r-2,c+1), (r-1,c+1), (r,c+1), (r-1,c+2)),
                                                ((r,c), (r-1,c+1), (r,c+1), (r+1,c+1), (r+1,c+2)),
                                                ((r,c), (r,c+1), (r+1,c+1), (r+2,c+1), (r+1,c+2)),
                                                ((r,c), (r-1,c+1), (r,c+1), (r+1,c+1), (r-1,c+2)),])

"""
*
* *
* *

  *
* *
* *

  * *
* * *

* *
* * *

* * *
* *

* * *
  * *

* *
* *
*

* *
* *
  *
"""
P_shape = shape_spec(name='P',
                     isomorphisms=lambda r, c: [((r,c), (r+1,c), (r+2,c), (r+1,c+1), (r+2,c+1)),
                                                ((r,c), (r+1,c), (r-1,c+1), (r,c+1), (r+1,c+1)),
                                                ((r,c), (r-1,c+1), (r,c+1), (r-1,c+2), (r,c+2)),
                                                ((r,c), (r+1,c), (r,c+1), (r+1,c+1), (r+1,c+2)),
                                                ((r,c), (r+1,c), (r,c+1), (r+1,c+1), (r,c+2)),
                                                ((r,c), (r,c+1), (r+1,c+1), (r,c+2), (r+1,c+2)),
                                                ((r,c), (r+1,c), (r+2,c), (r,c+1), (r+1,c+1)),
                                                ((r,c), (r+1,c), (r,c+1), (r+1,c+1), (r+2,c+1)),])

pentominoes =  [X_shape, I_shape, V_shape, U_shape, W_shape, T_shape,
                Z_shape, N_shape, L_shape, Y_shape, F_shape, P_shape]

# Fibonacci's tilings __________________________________________________

"""
*
"""
square_shape = shape_spec(
    name='square',
    isomorphisms=lambda r, c: [((r, c),)])

"""
* *    *
       *
"""
domino_shape = shape_spec(
    name='domino',
    isomorphisms=lambda r, c: [ ((r, c), (r, c+1)),
                                ((r, c), (r+1, c))])

fibonacci_shapes = [square_shape, domino_shape]


def parallelogram_polyominoes(sp):
    """
    Returns the set of shapes denoting all *parallelogram polyominoes* of semiperimeter `sp`.

    The relation to the main problem is $sp = n + 2$, where the board edge is $2^{n}$.
    """

    import itertools

    steps = [(1,0),  # go to cell below
             (0,1),] # go to cell on the right

    # we assume the canonical ordering inside a board, namely ascending
    # order from top to bottom, and from left to right, as in cells above.

    # first of all build Catalan paths that have no intersection point,
    # using only vertical and horizontal steps (here we do not distinguish
    # among upward/downward vertical steps, what is important is to remain
    # consistent to board ordering).

    initial_prefix = [(0,0)] # always start placing cell at anchor coordinate (r, c)
    prefixes = {tuple(initial_prefix)}

    candidates = set()

    n = sp - 2

    while prefixes:

        prefix = prefixes.pop()

        if len(prefix) > n:
            candidates.add(prefix)
            continue

        last_row_offset, last_col_offset = prefix[-1]
        dominating_prefixes = [(last_row_offset + ro, last_col_offset + co)
                               for ro, co in steps]
        dominated_prefixes  = [(last_row_offset + ro, last_col_offset + co)
                               for ro, co in steps]
        for sub, dom in [(sub,dom)
                         for sub, dom
                         in zip(dominated_prefixes, dominating_prefixes)
                         if sub <= dom]: # rewrite this with `filter`
            prefixes.add(prefix + (sub,))
            prefixes.add(prefix + (dom,))

    parallelograms = {(tuple(dom), tuple(sub))
                      for sub, dom in itertools.product(candidates, repeat=2)
                      if sub[-1] == dom[-1]}

    polyominoes = {frozenset(fst) | frozenset(snd)
                   for fst, snd in parallelograms}

    def fill(polyomino):
        r_max, c_max = max(r for r,_ in polyomino), max(c for _,c in polyomino)
        filled = set()
        for coord in itertools.product(range(1, r_max), range(1, c_max)):

            if coord in polyomino: continue

            row = [(rr, cc) for rr, cc in polyomino if coord[0] == rr]
            column = [(rr, cc) for rr, cc in polyomino if coord[1] == cc]

            if (min(column, default=coord) < coord < max(column, default=coord)
                and min(row, default=coord) < coord < max(row, default=coord)):

                filled.add(coord)

        return polyomino | filled

    return {fill(p) for p in polyominoes}

def describe(semiperimeter, polyominoes):

    n, catalan = semiperimeter - 2, semiperimeter - 1
    size=2**n

    area = 0
    for p in polyominoes:
        area += len(p)

    theoretical_area = 4**n

    code='''
    **Problem instance**:
    - using {pp} parallelogram polyominoes, which is the catalan number $c_{catalan}$ (0-based indexing)
    - each polyomino has semiperimeter $sp$ equals to {sp}, so let $n=sp-2={n}$
    - board edges have size $2^{n}={size}$ each
    - theoretically, area is known to be $4^{n}={fa}$, which *is {pred}* equal to counted area {ca}
    '''.format(
        sp=semiperimeter, n=n, size=size, pp=len(polyominoes), fa=theoretical_area,
        pred='not' if theoretical_area != area else '', ca=area, catalan=catalan)

    return code, n, catalan, polyominoes, theoretical_area, area, size

# ______________________________________________________________________________

def rotate_clockwise(shape):
    """
    Returns a shape rotated clockwise by π/2 radians.
    """
    clockwise = [(c, -r) for r, c in shape] # by matrix multiplication for rotations
    #print(clockwise)
    co, ro = min((c, r) for r, c in clockwise)
    #print(ro, co)
    return frozenset((r-ro, c-co) for r, c in clockwise)

def rotate_clockwise_isomorphisms(isos):

    rotations = set()

    for i in isos:
        rotating = i

        rotating = rotate_clockwise(rotating)
        rotations.add(rotating)

        rotating = rotate_clockwise(rotating)
        rotations.add(rotating)

        rotating = rotate_clockwise(rotating)
        rotations.add(rotating)

    #print(isos | frozenset(rotations))
    return isos | rotations

#______________________________________________________________________________

def vertical_mirror(shape):
    m = max([c for r, c in shape])
    return frozenset((r, m-c) for r, c in shape)

def vertical_isomorphism(isos):
    return isos | frozenset(vertical_mirror(i) for i in isos)

#______________________________________________________________________________

def make_shapes(primitive_polyos, isomorphisms=lambda isos: isos):

    prefix = "polyo"

    def o(j, isos):
        return shape_spec(name='{}{}'.format(prefix, j),
                          isomorphisms=lambda r, c: [[(r+ro, c+co) for ro, co in iso]
                                                     for iso in isos])

    return [o(j, isomorphisms({p})) for j, p in enumerate(primitive_polyos)]


def convex_hull(iso):
    lower_row, lower_col = min(iso)
    greatest_row, greatest_col = max(iso) # find the "convex hull" respect to the cell at the bottom-right most location
    return [(r, c)
            for r in range(lower_row, greatest_row+1)
            for c in range(lower_col, greatest_col+1)] # convex hull area

def area_key(s, semiperimeter,
             weights={'random':0,
                      'max row':0,
                      'max col':0,
                      'filled area':1,
                      'convex-hull ratio':1}):
    import random
    isos = s.isomorphisms(0,0) # place the shape somewhere, the origin is pretty good
    iso = isos.pop() # all isomorphisms, of course, have the same are, therefore pick the first one
    filled_area = len(iso) # since a piece is represent as an iterable of coordinates `(r, c)`
    convex_hull_area = len(convex_hull(iso))

    def key(x):
        max_r, max_c = max(iso)
        max_convex_hull = (semiperimeter-1)**2
        #return max_r + max_c*x + (filled_area * convex_hull_area / max_convex_hull)*x**2
        return (weights['random']*random.random()*x**0 +
                weights['max row']*max_r*x**1 +
                weights['max col']*max_c*x**2 +
                weights['filled area']*filled_area*x**3 +
                weights['convex-hull ratio']*(convex_hull_area / max_convex_hull)*x**4)

    return key(1)


def not_insertion_on_edges(coord, positions, shapes, size):

    r, c = coord

    return r >= size-1 or c >= size-1

    if r >= size-1 or c >= size-1:
        return True

    return False

    for s in shapes:
        iso = s.isomorphisms(r, c).pop()
        max_r, max_c = max(iso)
        if max_r < size and max_c < size:
            break
    else:
        return True

    return False


# ________________________________________________________________________________

def doctests():
    '''
    Doctests, simply.

    >>> dim = (6,10)
    >>> polys_sols = polyominoes(dim, pentominoes, availables="ones", forbidden=[])
    >>> tilings = markdown_pretty(polys_sols, dim, pentominoes, pretty_printer=symbols_pretty, raw_text=True)
    >>> for i in range(6): # doctest: +NORMALIZE_WHITESPACE
    ...     print(next(tilings))
    ┌─────────────────────┐
    │ β δ δ δ ε ε ι ι ι ι │
    │ β δ θ δ α ε ε λ λ ι │
    │ β θ θ α α α ε η λ λ │
    │ β θ γ μ α η η η λ ζ │
    │ β θ γ μ μ η κ ζ ζ ζ │
    │ γ γ γ μ μ κ κ κ κ ζ │
    └─────────────────────┘
    ┌─────────────────────┐
    │ β δ δ δ η η α ζ ζ ζ │
    │ β δ θ δ η α α α ζ κ │
    │ β θ θ η η λ α ε ζ κ │
    │ β θ γ λ λ λ ε ε κ κ │
    │ β θ γ ι λ ε ε μ μ κ │
    │ γ γ γ ι ι ι ι μ μ μ │
    └─────────────────────┘
    ┌─────────────────────┐
    │ β δ δ δ η η ι ι ι ι │
    │ β δ θ δ η ε ε λ λ ι │
    │ β θ θ η η α ε ε λ λ │
    │ β θ γ μ α α α ε λ ζ │
    │ β θ γ μ μ α κ ζ ζ ζ │
    │ γ γ γ μ μ κ κ κ κ ζ │
    └─────────────────────┘
    ┌─────────────────────┐
    │ β ε ε ζ ζ ζ ι ι ι ι │
    │ β κ ε ε ζ λ θ θ θ ι │
    │ β κ κ ε ζ λ λ λ θ θ │
    │ β κ γ δ δ α λ η η μ │
    │ β κ γ δ α α α η μ μ │
    │ γ γ γ δ δ α η η μ μ │
    └─────────────────────┘
    ┌─────────────────────┐
    │ β ε ε ζ ζ ζ ι ι ι ι │
    │ β κ ε ε ζ λ θ θ θ ι │
    │ β κ κ ε ζ λ λ λ θ θ │
    │ β κ γ μ η η λ α δ δ │
    │ β κ γ μ μ η α α α δ │
    │ γ γ γ μ μ η η α δ δ │
    └─────────────────────┘
    ┌─────────────────────┐
    │ β ε ε μ μ μ ζ δ δ δ │
    │ β κ ε ε μ μ ζ δ θ δ │
    │ β κ κ ε α ζ ζ ζ θ θ │
    │ β κ γ α α α λ η η θ │
    │ β κ γ ι α λ λ λ η θ │
    │ γ γ γ ι ι ι ι λ η η │
    └─────────────────────┘

    >>> dim = (6, 10)
    >>> polys_sols = polyominoes(
    ...                 dim, pentominoes,
    ...                 availables="ones",
    ...                 forbidden=[(0,0), (1,0), (2,0), (3,0), (4,0),
    ...                            (1,9),(2,9),(3,9),(4,9), (5,9),
    ...                            (1,5), (2, 4), (2, 5), (3, 4), (3,5)])
    >>> tilings = markdown_pretty(polys_sols, dim, pentominoes, pretty_printer=symbols_pretty, raw_text=True)
    >>> for i in range(6): # doctest: +NORMALIZE_WHITESPACE
    ...     print(next(tilings))
    ┌─────────────────────┐
    │   γ γ γ δ δ δ ζ ζ ζ │
    │   ι ι γ δ   δ λ ζ   │
    │   ι μ γ     λ λ ζ   │
    │   ι μ μ     η λ λ   │
    │   ι μ μ θ θ η η η   │
    │ β β β β β θ θ θ η   │
    └─────────────────────┘
    ┌─────────────────────┐
    │   γ γ γ ι ι ι ι λ λ │
    │   μ μ γ ι   ε λ λ   │
    │   μ μ γ     ε ε λ   │
    │   δ μ δ     η ε ε   │
    │   δ δ δ θ θ η η η   │
    │ β β β β β θ θ θ η   │
    └─────────────────────┘
    ┌─────────────────────┐
    │   γ γ γ ι ι ι ι λ λ │
    │   γ μ μ ι   ε λ λ   │
    │   γ μ μ     ε ε λ   │
    │   δ μ δ     η ε ε   │
    │   δ δ δ θ θ η η η   │
    │ β β β β β θ θ θ η   │
    └─────────────────────┘
    ┌─────────────────────┐
    │   γ γ γ θ θ δ δ λ λ │
    │   γ θ θ θ   δ λ λ   │
    │   γ ε ε     δ δ λ   │
    │   ε ε κ     ζ μ μ   │
    │   ε κ κ κ κ ζ μ μ   │
    │ β β β β β ζ ζ ζ μ   │
    └─────────────────────┘
    ┌─────────────────────┐
    │   γ γ γ θ θ δ δ λ λ │
    │   γ θ θ θ   δ λ λ   │
    │   γ ε ε     δ δ λ   │
    │   ε ε κ     η η μ   │
    │   ε κ κ κ κ η μ μ   │
    │ β β β β β η η μ μ   │
    └─────────────────────┘
    ┌─────────────────────┐
    │   γ γ γ θ θ δ δ λ λ │
    │   γ θ θ θ   δ λ λ   │
    │   γ ε ε     δ δ λ   │
    │   ε ε κ     μ μ μ   │
    │   ε κ κ κ κ μ μ ι   │
    │ β β β β β ι ι ι ι   │
    └─────────────────────┘

    >>> repeated_Y_shapes = [shape_spec(name="{}_{}".format(Y_shape.name, i),
    ...                                 isomorphisms=Y_shape.isomorphisms)
    ...                                 for i in range(10)]
    >>> dim = (5,10)
    >>> polys_sols = polyominoes(dim, repeated_Y_shapes, availables='ones')
    >>> tilings = markdown_pretty(polys_sols, dim, repeated_Y_shapes, pretty_printer=symbols_pretty, raw_text=True)
    >>> for i in range(4): # doctest: +NORMALIZE_WHITESPACE
    ...     print(next(tilings))
    ┌─────────────────────┐
    │ α γ γ γ γ ζ ι ι ι ι │
    │ α α δ γ ζ ζ ζ ζ ι κ │
    │ α δ δ δ δ η η η η κ │
    │ α β ε ε ε ε θ η κ κ │
    │ β β β β ε θ θ θ θ κ │
    └─────────────────────┘
    ┌─────────────────────┐
    │ α γ γ γ γ ζ κ κ κ κ │
    │ α α δ γ ζ ζ ζ ζ κ ι │
    │ α δ δ δ δ η η η η ι │
    │ α β ε ε ε ε θ η ι ι │
    │ β β β β ε θ θ θ θ ι │
    └─────────────────────┘
    ┌─────────────────────┐
    │ α γ γ γ γ ζ θ θ θ θ │
    │ α α δ γ ζ ζ ζ ζ θ κ │
    │ α δ δ δ δ η η η η κ │
    │ α β ε ε ε ε ι η κ κ │
    │ β β β β ε ι ι ι ι κ │
    └─────────────────────┘
    ┌─────────────────────┐
    │ α γ γ γ γ ζ κ κ κ κ │
    │ α α δ γ ζ ζ ζ ζ κ θ │
    │ α δ δ δ δ η η η η θ │
    │ α β ε ε ε ε ι η θ θ │
    │ β β β β ε ι ι ι ι θ │
    └─────────────────────┘


    >>> [ [len(list(polyominoes(dim=(j,i), shapes=fibonacci_shapes, availables='inf'))) # doctest: +NORMALIZE_WHITESPACE
    ...    for i in range(n)]
    ...  for j, n in [(1, 13),  # https://oeis.org/A000045
    ...               (2, 13),  # https://oeis.org/A030186
    ...               (3, 7),   # https://oeis.org/A033506
    ...               (4, 7),   # https://oeis.org/A033507
    ...               (5, 6)]]  # https://oeis.org/A033508
    [[0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233],
     [0, 2, 7, 22, 71, 228, 733, 2356, 7573, 24342, 78243, 251498, 808395],
     [0, 3, 22, 131, 823, 5096, 31687],
     [0, 5, 71, 823, 10012, 120465, 1453535],
     [0, 8, 228, 5096, 120465, 2810694]]


    >>> dim = (1,6)
    >>> fibs_sols = polyominoes(dim, fibonacci_shapes, availables='inf')
    >>> fibs_tilings = list(markdown_pretty(fibs_sols, dim, fibonacci_shapes, pretty_printer=symbols_pretty, raw_text=True))
    >>> for t in fibs_tilings: # doctest: +NORMALIZE_WHITESPACE
    ...     print(t)
    ┌─────────────┐
    │ α α α α α α │
    └─────────────┘
    ┌─────────────┐
    │ α α α α β β │
    └─────────────┘
    ┌─────────────┐
    │ α α α β β α │
    └─────────────┘
    ┌─────────────┐
    │ α α β β α α │
    └─────────────┘
    ┌─────────────┐
    │ α α β β β β │
    └─────────────┘
    ┌─────────────┐
    │ α β β α α α │
    └─────────────┘
    ┌─────────────┐
    │ α β β α β β │
    └─────────────┘
    ┌─────────────┐
    │ α β β β β α │
    └─────────────┘
    ┌─────────────┐
    │ β β α α α α │
    └─────────────┘
    ┌─────────────┐
    │ β β α α β β │
    └─────────────┘
    ┌─────────────┐
    │ β β α β β α │
    └─────────────┘
    ┌─────────────┐
    │ β β β β α α │
    └─────────────┘
    ┌─────────────┐
    │ β β β β β β │
    └─────────────┘


    >>> semiperimeter = 6
    >>> description = describe(semiperimeter, parallelogram_polyominoes(semiperimeter))
    >>> statement, n, catalan, polyos, theoretical_area, area, size = description
    >>> print(statement) # doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    **Problem instance**:
        - using 42 parallelogram polyominoes, which is the catalan number $c_5$ (0-based indexing)
        - each polyomino has semiperimeter $sp$ equals to 6, so let $n=sp-2=4$
        - board edges have size $2^4=16$ each
        - theoretically, area is known to be $4^4=256$, which *is * equal to counted area 256
    <BLANKLINE>

    >>> #shapes = make_shapes(pp, isomorphisms=vertical_isomorphism)#, isomorphisms=lambda isos: vertical_isomorphism(rotate_clockwise_isomorphisms(isos)))
    >>> parallelogram_polyominoes = make_shapes(polyos)#, isomorphisms=lambda isos: vertical_isomorphism(rotate_clockwise_isomorphisms(isos)))

    >>> parallelogram_polyominoes = sorted(parallelogram_polyominoes,
    ...                 key=functools.partial(area_key, semiperimeter=semiperimeter),
    ...                 reverse=True)

    >>> polyominoes_boards = [
    ...     symbols_pretty([(shape, None, (0,0), iso)],
    ...                     (semiperimeter, semiperimeter),
    ...                     {shape.name:'▢'},
    ...                     joiner=lambda board: board,
    ...                     axis=True)
    ...     for i, shape in enumerate(parallelogram_polyominoes)
    ...     for iso in shape.isomorphisms(0,0)]

    >>> cols = 6
    >>> for i in range(0, len(polyominoes_boards), cols): # doctest: +NORMALIZE_WHITESPACE
    ...     for k in range(semiperimeter+2):
    ...         grouped_board = []
    ...         for j in range(i, i + cols):
    ...             if j >= len(polyominoes_boards): break
    ...             grouped_board.append(polyominoes_boards[j][k])
    ...         print(' '.join(grouped_board))
    ┌─────────────> ┌─────────────> ┌─────────────> ┌─────────────> ┌─────────────> ┌─────────────>
    │ ▢ ▢ ▢         │ ▢ ▢ ▢         │ ▢ ▢           │ ▢ ▢           │ ▢ ▢ ▢ ▢       │ ▢ ▢ ▢
    │ ▢ ▢ ▢         │ ▢ ▢ ▢         │ ▢ ▢ ▢         │ ▢ ▢           │ ▢ ▢ ▢ ▢       │ ▢ ▢ ▢
    │ ▢ ▢ ▢         │   ▢ ▢         │ ▢ ▢ ▢         │ ▢ ▢           │               │     ▢
    │               │               │               │ ▢ ▢           │               │
    │               │               │               │               │               │
    │               │               │               │               │               │
    v               v               v               v               v               v
    ┌─────────────> ┌─────────────> ┌─────────────> ┌─────────────> ┌─────────────> ┌─────────────>
    │ ▢ ▢ ▢         │ ▢ ▢           │ ▢ ▢           │ ▢             │ ▢             │ ▢ ▢
    │   ▢ ▢         │ ▢ ▢           │ ▢ ▢ ▢         │ ▢ ▢ ▢         │ ▢ ▢           │ ▢ ▢
    │   ▢ ▢         │ ▢ ▢ ▢         │   ▢ ▢         │ ▢ ▢ ▢         │ ▢ ▢           │ ▢ ▢
    │               │               │               │               │ ▢ ▢           │   ▢
    │               │               │               │               │               │
    │               │               │               │               │               │
    v               v               v               v               v               v
    ┌─────────────> ┌─────────────> ┌─────────────> ┌─────────────> ┌─────────────> ┌─────────────>
    │ ▢ ▢ ▢ ▢       │ ▢ ▢ ▢         │ ▢ ▢           │ ▢ ▢ ▢         │ ▢ ▢           │ ▢
    │   ▢ ▢ ▢       │ ▢ ▢ ▢ ▢       │ ▢ ▢           │   ▢ ▢         │ ▢ ▢ ▢         │ ▢ ▢
    │               │               │   ▢ ▢         │     ▢         │     ▢         │ ▢ ▢ ▢
    │               │               │               │               │               │
    │               │               │               │               │               │
    │               │               │               │               │               │
    v               v               v               v               v               v
    ┌─────────────> ┌─────────────> ┌─────────────> ┌─────────────> ┌─────────────> ┌─────────────>
    │ ▢             │ ▢ ▢           │ ▢             │ ▢ ▢ ▢         │ ▢ ▢           │ ▢ ▢ ▢ ▢
    │ ▢ ▢ ▢         │   ▢ ▢         │ ▢ ▢           │   ▢ ▢ ▢       │ ▢ ▢ ▢ ▢       │     ▢ ▢
    │   ▢ ▢         │   ▢ ▢         │ ▢ ▢           │               │               │
    │               │               │   ▢           │               │               │
    │               │               │               │               │               │
    │               │               │               │               │               │
    v               v               v               v               v               v
    ┌─────────────> ┌─────────────> ┌─────────────> ┌─────────────> ┌─────────────> ┌─────────────>
    │ ▢ ▢           │ ▢             │ ▢ ▢ ▢         │ ▢             │ ▢             │ ▢ ▢
    │ ▢ ▢           │ ▢             │     ▢         │ ▢ ▢           │ ▢ ▢ ▢         │   ▢
    │   ▢           │ ▢ ▢           │     ▢         │   ▢ ▢         │     ▢         │   ▢ ▢
    │   ▢           │ ▢ ▢           │               │               │               │
    │               │               │               │               │               │
    │               │               │               │               │               │
    v               v               v               v               v               v
    ┌─────────────> ┌─────────────> ┌─────────────> ┌─────────────> ┌─────────────> ┌─────────────>
    │ ▢ ▢           │ ▢             │ ▢             │ ▢ ▢           │ ▢             │ ▢ ▢ ▢
    │   ▢ ▢         │ ▢             │ ▢ ▢ ▢ ▢       │   ▢           │ ▢             │     ▢ ▢
    │     ▢         │ ▢ ▢ ▢         │               │   ▢           │ ▢             │
    │               │               │               │   ▢           │ ▢ ▢           │
    │               │               │               │               │               │
    │               │               │               │               │               │
    v               v               v               v               v               v
    ┌─────────────> ┌─────────────> ┌─────────────> ┌─────────────> ┌─────────────> ┌─────────────>
    │ ▢             │ ▢ ▢ ▢ ▢       │ ▢             │ ▢ ▢           │ ▢             │ ▢ ▢ ▢ ▢ ▢
    │ ▢             │       ▢       │ ▢ ▢           │   ▢ ▢ ▢       │ ▢             │
    │ ▢ ▢           │               │   ▢           │               │ ▢             │
    │   ▢           │               │   ▢           │               │ ▢             │
    │               │               │               │               │ ▢             │
    │               │               │               │               │               │
    v               v               v               v               v               v

    >>> dim = (size, size)
    >>> polys_sols = polyominoes(dim, parallelogram_polyominoes, availables="ones",
    ...                          max_depth_reached=40,
    ...                          pruning=functools.partial(not_insertion_on_edges,
    ...                                                    size=size))
    >>> pretty_tilings = markdown_pretty(polys_sols, dim, parallelogram_polyominoes, raw_text=True)
    >>> print(next(pretty_tilings)) # doctest: +NORMALIZE_WHITESPACE
     _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    |     |   | |_    |_ _ _ _ _| |●
    |     |   |   |_  |   |   | |   |
    |_ _ _|   |   | |_|_  |   | |_  |
    |     |_ _|_ _|_    | |_  | | |_|
    |_    |     |   |_ _|_| |_| | |●
    | |_ _|_ _  |_  |_    |_  |_|   |
    |     |   |_| |_ _|_ _ _|_ _|_ _|
    |_ _ _|_ _  |   |_  |_  |_  | |●
    |   |_    |_|_ _ _|_  |   | |_  |
    |     |   |     |_  |_|_ _|_ _| |
    |_ _ _|_ _|_ _ _ _|_ _ _|_  | |_|
    |       |_      |   |_ _  | | |●
    |_ _ _ _| |_ _ _|_ _ _ _| | | |●
    |   |   |_ _  |_ _ _  | |_|_|_ _|
    |   |_    | |_|_ _  |_| |_ _    |
    |_ _ _|_ _|_ _ _ _|_ _|_ _ _|_ _|

    '''


