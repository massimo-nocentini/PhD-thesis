from bits import *

def queens(n):

    sol = [0] * n                                                   # Initialize the permutation that has *columns*
                                                                    # indices and *rows* as values.
    def gen(c, rows, raises, falls):

        for r in range(n):                                          # For each row index:

            raising = c + r                                         # when r > c negative positions appear in the
            falling = (c - r) % (2*n-1)                             # *most significant part* of `falling`.

            if (is_on(rows, r)                                      # if there is no queen on the same row and on
                and is_on(raises, raising)                          # the same raising and falling diagonals,
                and is_on(falls, falling)):                         # then `r,c` is a candidate position.

                sol[c] = r                                          # remember the choice of `r`.

                if c == n-1:                                        # if this recursive call concerns the last column
                    yield sol                                       # then no more work has to be done, yield a result
                else:
                    yield from gen(c+1,                             # otherwise, recur looking for a location in the
                                   clear_bit(rows, r),              # remaining columns, propagating the fact that 
                                   clear_bit(raises, raising),      # row `r` and diagonals `raising` and `falling`
                                   clear_bit(falls, falling))       # are under attack and no more selectable.

    return gen(0, set_all(n), set_all(2*n-1), set_all(2*n-1))       # start the enumeration of solutions.

def pretty(sol):
    n = len(sol)
    s = ""
    for r in range(n):
        pos = sol.index(r)
        row = "|".join('Q' if c == pos else ' '
                       for c in range(n))
        s += "|{}|\n".format(row)

    return s

#_______________________________________________________________________

def doctests():
    '''
    Doctests, simply.

    >>> import bits

    >>> for s in queens(5): # doctest: +NORMALIZE_WHITESPACE
    ...     print(pretty(s))
    |Q| | | | |
    | | | |Q| |
    | |Q| | | |
    | | | | |Q|
    | | |Q| | |
    <BLANKLINE>
    |Q| | | | |
    | | |Q| | |
    | | | | |Q|
    | |Q| | | |
    | | | |Q| |
    <BLANKLINE>
    | | |Q| | |
    |Q| | | | |
    | | | |Q| |
    | |Q| | | |
    | | | | |Q|
    <BLANKLINE>
    | | | |Q| |
    |Q| | | | |
    | | |Q| | |
    | | | | |Q|
    | |Q| | | |
    <BLANKLINE>
    | |Q| | | |
    | | | |Q| |
    |Q| | | | |
    | | |Q| | |
    | | | | |Q|
    <BLANKLINE>
    | | | | |Q|
    | | |Q| | |
    |Q| | | | |
    | | | |Q| |
    | |Q| | | |
    <BLANKLINE>
    | |Q| | | |
    | | | | |Q|
    | | |Q| | |
    |Q| | | | |
    | | | |Q| |
    <BLANKLINE>
    | | | | |Q|
    | |Q| | | |
    | | | |Q| |
    |Q| | | | |
    | | |Q| | |
    <BLANKLINE>
    | | | |Q| |
    | |Q| | | |
    | | | | |Q|
    | | |Q| | |
    |Q| | | | |
    <BLANKLINE>
    | | |Q| | |
    | | | | |Q|
    | |Q| | | |
    | | | |Q| |
    |Q| | | | |
    <BLANKLINE>


    >>> [len(list(queens(i))) for i in range(1,13)]
    [1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200]


    >>> more_queens = queens(24)
    >>> print(pretty(next(more_queens))) # doctest: +NORMALIZE_WHITESPACE
    |Q| | | | | | | | | | | | | | | | | | | | | | | |
    | | | |Q| | | | | | | | | | | | | | | | | | | | |
    | |Q| | | | | | | | | | | | | | | | | | | | | | |
    | | | | |Q| | | | | | | | | | | | | | | | | | | |
    | | |Q| | | | | | | | | | | | | | | | | | | | | |
    | | | | | | | | | | | | | | | | |Q| | | | | | | |
    | | | | | | | | | | | | | | | | | | | | | |Q| | |
    | | | | | | | | | | | | | | | | | |Q| | | | | | |
    | | | | | |Q| | | | | | | | | | | | | | | | | | |
    | | | | | | | | | | | | | | |Q| | | | | | | | | |
    | | | | | | |Q| | | | | | | | | | | | | | | | | |
    | | | | | | | | | | | | | | | | | | |Q| | | | | |
    | | | | | | | | | | | | | | | | | | | | |Q| | | |
    | | | | | | | |Q| | | | | | | | | | | | | | | | |
    | | | | | | | | | | | | | | | | | | | | | | | |Q|
    | | | | | | | | | | | | | | | | | | | |Q| | | | |
    | | | | | | | | | | | | | | | | | | | | | | |Q| |
    | | | | | | | | |Q| | | | | | | | | | | | | | | |
    | | | | | | | | | | |Q| | | | | | | | | | | | | |
    | | | | | | | | | | | | |Q| | | | | | | | | | | |
    | | | | | | | | | | | | | | | |Q| | | | | | | | |
    | | | | | | | | | |Q| | | | | | | | | | | | | | |
    | | | | | | | | | | | |Q| | | | | | | | | | | | |
    | | | | | | | | | | | | | |Q| | | | | | | | | | |
    <BLANKLINE>


    '''
    pass

