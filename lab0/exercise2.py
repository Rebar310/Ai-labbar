
# Write a function which gets two numbers, namely b as the base and a as the exponent. The function must return
# the result of raising b to the power of a, i.e. b^a


def power(b,a):
    if a == 0:
        return 1

    result = 1
    exp = abs(a)

    for _ in range(exp):
        result *= b

    if a < 0:
        return 1 / result

    return result


assert power(1, 500) == 1
assert power(3, 4) == 81
assert power(0, 10) == 0
assert power(-2, 3) == -8
assert power(5, -2) == 0.04
assert power(0, 0) == 1








