from bits import *

def queens(n):

    sol = [0] * n

    def gen(c, rows, raises, falls):

        for r in range(n):

            raising = c + r

            # we use a modular ring in order to handle the case
            # r > c, in this way negative positions appear in the
            # most significant part of `falls`.
            falling = (c - r) % (2*n-1)


            if (is_on(rows, r)
                and is_on(raises, raising)
                and is_on(falls, falling)):

                sol[c] = r

                if c == n-1:
                    yield sol
                else:
                    yield from gen(c+1,
                                   clear_bit(rows, r),
                                   clear_bit(raises, raising),
                                   clear_bit(falls, falling))

    return gen(0, set_all(n), set_all(2*n-1), set_all(2*n-1))

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

