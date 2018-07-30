
def doctests():
    """
    Doctests, simply.

    >>> from sympy import Symbol
    >>> a_sym = Symbol('a')
    >>> a_sym
    a

    >>> from sympy import Eq, solve, symbols
    >>> a, t = symbols('a t')
    >>> a_def = Eq(a, 3)
    >>> at_eq = Eq(a+5*t, 1/(1-t))
    >>> sols = [Eq(t, s) for s in solve(at_eq, t)]
    >>> a_def, at_eq, sols # doctest: +NORMALIZE_WHITESPACE
    (Eq(a, 3),
     Eq(a + 5*t, 1/(-t + 1)),
     [Eq(t, -a/10 - sqrt(a**2 + 10*a + 5)/10 + 1/2),
      Eq(t, -a/10 + sqrt(a**2 + 10*a + 5)/10 + 1/2)])


    >>> from commons import define
    >>> from sympy import Function, sqrt
    >>> f = Function('f')
    >>> f(3)
    f(3)
    >>> t = symbols('t')
    >>> define(let=f(t), be=(1-sqrt(1-4*t))/(2*t))
    Eq(f(t), (-sqrt(-4*t + 1) + 1)/(2*t))

    >>> from sympy import latex
    >>> from commons import lift_to_Lambda
    >>> from sympy import IndexedBase
    >>> n = symbols('n')
    >>> a = IndexedBase('a')
    >>> aeq = Eq(a[n], n+a[n-1])
    >>> with lift_to_Lambda(aeq, return_eq=True) as aEQ:
    ...     arec = aEQ(n+1)
    >>> arec
    Eq(a[n + 1], n + a[n] + 1)
    >>> print(latex(arec))
    a_{n + 1} = n + a_{n} + 1

    >>> b = Function('b')
    >>> beq = Eq(b(n), n+b(n-1))
    >>> with lift_to_Lambda(beq, return_eq=True) as bEQ:
    ...     brec = bEQ(n+1)
    >>> brec
    Eq(b(n + 1), n + b(n) + 1)
    >>> print(latex(brec))
    b{\\left (n + 1 \\right )} = n + b{\\left (n \\right )} + 1

    """
    pass

