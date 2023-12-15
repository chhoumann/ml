""" For calc_it """


def calc_it(st_: str) -> int:
    """Calculates the decimal value for the given hex value"""
    st_ = st_.replace("0x", "")
    mapped = {"F": 15, "E": 14, "D": 13, "C": 12, "B": 11, "A": 10}
    summed = 0
    for char, i in zip(st_, range(len(st_), 0, -1)):
        ch_ = char.capitalize()
        summed += (mapped[ch_] if mapped.get(ch_, False)
                   else int(char)) * 16 ** (i - 1)

    return summed
