


def hol_is_in_range(hol, d_min, d_max) -> bool:
    if hol >= d_min and hol <= d_max:
        return True
    else:
        return False


def hol_flat_reward(hol, d_min, d_max) -> float:
    if not hol_is_in_range(hol, d_min, d_max):
        return 0.
    else:
        return 1.


if __name__ == '__main__':

    a = hol_is_in_range(7, 4, 6)
    print(a)

    a = hol_flat_reward(2, 4, 60)
    print(a)
