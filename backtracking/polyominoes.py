
import math, random
from bits import *
from collections import namedtuple

shape_spec = namedtuple('shape_spec', ['name', 'isomorphisms',])

def polyominoes(dim, shapes, availables='ones',
                max_depth_reached=None, forbidden=[],):

    rows, cols = dim
    sol = []

    if not availables or availables == 'ones':
        availables = {s.name:1 for s in shapes}
    elif availables == 'inf':
        # trick: represent ∞ piece availability
        # by decreasing negative numbers
        availables = {s.name:-1 for s in shapes}

    def place(S, positions):
        for r, c in positions:
            S = clear_bit(S, r + rows*c)
        return S

    def gen(positions, attempts):

        p = low_bit(positions)

        c, r = divmod(p, rows)

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
*      * * *      *  * * *
*          *      *  *
* * *      *  * * *  *
"""
V_shape = shape_spec(
            name='V', 
            isomorphisms=lambda r, c: [
                ((r,c), (r+1,c), (r+2,c), (r+2, c+1), (r+2, c+2)), 
                ((r,c), (r, c+1), (r,c+2), (r+1, c+2), (r+2, c+2)),
                ((r,c), (r,c+1), (r-2,c+2), (r-1,c+2), (r, c+2)),
                ((r,c), (r+1,c), (r+2,c), (r,c+1), (r, c+2))
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

shapes =  [X_shape, I_shape, V_shape, U_shape, W_shape, T_shape,
           Z_shape, N_shape, L_shape, Y_shape, F_shape, P_shape]

def doctests():
    '''
    Doctests, simply.

    >>> dim = (6,10)
    >>> polys_sols = polyominoes(dim, shapes, availables="ones", forbidden=[])
    >>> tilings = markdown_pretty(polys_sols, dim, shapes, pretty_printer=symbols_pretty, raw_text=True)
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


    with timing(lambda: len(list(polys_sols))) as (t, elapsed_time):
        print("{} sols in {} time".format(t, elapsed_time))
    '''
