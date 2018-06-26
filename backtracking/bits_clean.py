def is_on(S, j): 
    return (S & (1 << j)) >> j

def set_all(n):
    return (1 << n) - 1

def low_bit(S): 
    return (S & (-S)).bit_length() - 1

def clear_bit(S, j):
    return S & ~(1 << j)
