


def hol_is_in_range(hol, d_min, d_max) -> bool:
    if hol < d_min or hol > d_max:
        return False
    else:
        return True


def hol_flat_reward(hol, d_min, d_max) -> float:
    if hol_is_in_range(hol, d_min, d_max):
        return 5.
    else:
        return 0.


if __name__ == '__main__':

    a = hol_is_in_range(7, 4, 6)
    print(a)

    a = hol_flat_reward(2, 4, 60)
    print(a)
