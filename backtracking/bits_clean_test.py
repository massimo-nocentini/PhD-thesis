>>> S = 0b101010
>>> is_on(S, 3), is_on(S, 2)
(1, 0)

>>> bin(set_all(10))
'0b1111111111'

>>> low_bit(0b100101)
0
>>> low_bit(0b100100)
2
>>> low_bit(1 << 5)
5

>>> S = 0b101010
>>> bin(clear_bit(S, 1))
'0b101000'
