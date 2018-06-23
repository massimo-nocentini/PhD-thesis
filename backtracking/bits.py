

"""
The following set of functions allows us to use `int` objects as predicates,
namely they represent a set of boolean when we write them in binary.

Consider a predicate over six elements, true for those in position 1 and 5, 
namely the second and last elements, respectively.
>>> 0b100010
34

the other way, with `bin` we can take the set representation as a string object:
>>> bin(34)
'0b100010'

Multiply and divide by powers of 2, it is the same to insert and remove objects from 
the set or, in parallel, discard elements for which the predicate has no more sense:
>>> S = 0b100010
>>> S <<= 1
>>> S, bin(S)
(68, '0b1000100')
>>> S >>= 2
>>> S, bin(S)
(17, '0b10001')

"""


def from_mask(m, coding='big'):
    """
    Returns the integer represented by the bit mask `m`, given in `coding` scheme.

    Keyword argument `coding` represent the coding scheme used by mask `m`, it 
    resembles keyword `byteorder` in function `int.to_bytes`. If it is 'big', then
    a `big endian` scheme has been applied, namely bits at the *beginning* of `m` are the 
    most significant bits of the returned integer; otherwise a `little endian` scheme applies, 
    namely bits at the *end* of the given mask `m` are most significant bits of the
    returned integer.

    Examples
    ========

    >>> from_mask((1,1,0,0), coding='big')
    12

    >>> from_mask((0,0,1,1), coding='little')
    12

    """
    n = 0
    for i, b in enumerate(m if coding == 'little' else reversed(m)):
        n |= b * (1 << i)

    return n


def as_mask(n, coding='big'):
    """
    Returns `n` as a bit mask, namely an iterable of bits, according to `coding` scheme.

    Keyword argument `coding` represent the coding scheme used to build the mask, it 
    resembles keyword `byteorder` in function `int.to_bytes`. If it is 'big', then
    a `big endian` scheme is applied, namely most significant bits of `n` are at 
    the *beginning* of the returned mask; otherwise a `little endian` scheme applies, 
    namely most significant bits of `n` are at the *end* of the returned mask.

    Examples
    ========

    >>> as_mask(12, coding='big')
    (1, 1, 0, 0)

    >>> as_mask(12, coding='little')
    (0, 0, 1, 1)

    """
    m = [is_on(n, i) for i in range(n.bit_length())]
    return tuple(m if coding == 'little' else reversed(m))

def pretty_mask(m, coding='big', width=None):
    """
    Return a `str` object which is the pretty representation of mask `m`.

    Examples
    ========

    >>> pretty_mask(0b1100, coding='little', width=4)
    '0011'

    >>> pretty_mask(0b1100, coding='big', width=4)
    '1100'

    >>> pretty_mask(0b1101, coding='little', width=6)
    '101100'

    >>> pretty_mask(0b1101, coding='big', width=6)
    '001101'
    """
    if not width: 
        width = len(m)

    format_pattern = '{0:{fill}{align}{n}}'
    m_str = "".join(map(str, as_mask(m, coding)))
    return format_pattern.format(m_str, fill='0', 
                    align='<' if coding == 'little' else '>', n=width)

def ones_of(S):
    """
    Returns the positions of bits 1 in S, seen as a mask.

    The returned iterable can be interpreted as the set of 
    elements for which *predicate* `S` holds among `S.bit_length()`
    objects; eventually, it identifies a subset.

    Examples
    ========

    >>> ones_of(12)
    [2, 3]

    >>> ones_of(int(0b1000101010))
    [1, 3, 5, 9]

    """

    mask = as_mask(S, coding='little')
    return [i for i, m in enumerate(mask) if m]

def from_ones(ones):
    """
    Returns an integer with bits set to 1 according to positions in `ones`. 

    >>> bin(from_ones([1, 3, 5, 9]))
    '0b1000101010'

    """ 
    n = 0
    for o in ones:
        n = set_bit(n, o)
    return n
    
def subsets(n):
    """
    Returns an iterable of all subsets of a set composed of `n` objects.

    Subsets are generated by integer successors, therefore it is not
    known a priori the number of bits toggled between consecutive subsets.

    Examples
    ========

    >>> list(map(bin, subsets(4)))
    ['0b0', '0b1', '0b10', '0b11', '0b100', '0b101', '0b110', '0b111', '0b1000', '0b1001', '0b1010', '0b1011', '0b1100', '0b1101', '0b1110', '0b1111']

    """
    for i in range(1 << n):
        yield i
 
def subsets_of(S):
    """
    Returns all subsets of set `S`, given as an `int`.

    Examples
    ========

    >>> list(map(bin, subsets_of(0b1100)))
    ['0b1100', '0b1000', '0b100', '0b0']

    """
    x = S
    while x > 0:
        yield x
        x = S & (x-1) 

    yield x
       

def set_bit(S, j): 
    """
    Return a new set from set `S` with element in position `j` turned on.

    Examples
    ========

    Set/turn on the 3-th item of the set:
    >>> S = 0b100010
    >>> bin(set_bit(S, 3))
    '0b101010'

    """
    return S | (1 << j)

def is_on(S, j): 
    """
    Returns 1 if and only if the `j`-th item of the set `S` is on.

    Examples
    ========

    Check if the 3-th and then 2-nd item of the set is on:
    >>> S = 0b101010
    >>> is_on(S, 3), is_on(S, 2)
    (1, 0)

    """
    return (S & (1 << j)) >> j

def clear_bit(S, j):
    """
    Returns a new set from set `S` with the `j`-th item turned off.

    Examples
    ========

    Clear/turn off item in position 1 of the set:
    >>> S = 0b101010
    >>> bin(clear_bit(S, 1))
    '0b101000'

    """
    return S & ~(1 << j)

def low_bit_and_clear(S):
    """ 
    Returns a pair of `j = low_bit(S)` and `clear_bit(S, j)`.

    This is an aggregating function, similar in spirit to `test_and_set`,
    which consumes a set `S` and returns the position of the first bit 1,
    from the right (least significant part), and a new set with that bit 
    turned off.
    
    Examples
    ========

    >>> S = 0b101010
    >>> j, R = (low_bit_and_clear(S))
    >>> j
    1
    >>> bin(R)
    '0b101000'

    """
    j = low_bit(S)
    return j, clear_bit(S, j)

def toggle_bits(S, *positions):
    """
    Returns a new set from set `S` with bits toggled (flip the status of) in the given `positions`.

    Examples
    ========

    Toggle the 2-nd item and then 3-rd item of the set
    >>> S = int('0b101000', base=2)
    >>> S = toggle_bits(S, 2, 3)
    >>> bin(S)
    '0b100100'

    """
    for j in positions:
        S = S ^ (1 << j)

    return S

def low_bit(S): 
    return (S & (-S)).bit_length() - 1

def set_all(n):
    """
    Returns a new set composed by `n` elements.

    Examples
    ========

    Build a set with 10 elements within it:
    >>> bin(set_all(10))
    '0b1111111111'

    """
    return (1 << n) - 1

def modulo(S, N):
    '''
    Returns S % N, where N is a power of 2
    '''
    return S & (N - 1)

def is_power_of_two(S):
    return False if S & (S - 1) else True

def nearest_power_of_two(S):

    import math
    return math.floor(2**(math.log2(S) + .5))

def turn_off_last_bit(S):
    """
    Turns off the rightmost 1-bit in a word, producing 0 if none.

    By product, the position of toggled 1-bit is returned, -1 if none.

    Examples
    ========

    When a 1-bit is present:
    >>> S, b = turn_off_last_bit(0b1011000)
    >>> bin(S)
    '0b1010000'
    >>> b
    3

    When a 1-bit is the very least bit:
    >>> S, b = turn_off_last_bit(0b1)
    >>> bin(S)
    '0b0'
    >>> b
    0

    When a 1-bit is *not* present:
    >>> S, b = turn_off_last_bit(0b0)
    >>> bin(S)
    '0b0'
    >>> b
    -1

    """
    SS = S & (S - 1)
    b = (SS ^ S).bit_length() - 1
    return (SS, b) if S else (SS, -1)

def turn_on_last_zero(S): 
    """
    Turns on the rightmost 0-bit in word `S`, producing all 1's if none.

    By product, the position of toggled 0-bit is returned.

    Examples
    ========

    When a 0-bit is present:
    >>> S, z = turn_on_last_zero(0b10100111)
    >>> bin(S)
    '0b10101111'
    >>> z
    3

    When a 0-bit is *implicitly* present:
    >>> S, z = turn_on_last_zero(0b11111)
    >>> bin(S)
    '0b111111'
    >>> z
    5
    """
    SS = S | (S + 1)
    z = (SS ^ S).bit_length()-1
    return SS, z

def turn_off_last_consecutive_bits(S):
    return S & (S + 1)

def turn_on_last_consecutive_zeroes(S):
    return S | (S - 1)

#_______________________________________________________________________





