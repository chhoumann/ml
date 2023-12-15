import numpy as np

N = 30

if __name__ == "__main__":
    NO_BIRTHDAYS_WITHOUT_DUPES = 1
    for i in range(365, 365 - N, -1):
        NO_BIRTHDAYS_WITHOUT_DUPES *= i

    TOTAL_NO_N_BDAYS_NO_RESTRICTION = 365**N

    res = 1 - (NO_BIRTHDAYS_WITHOUT_DUPES / TOTAL_NO_N_BDAYS_NO_RESTRICTION)

    print(res)

# Here we are dealing with probabilities. This is called the birthday paradox.
# The formula goes:
# 1 - ((365 * 364 * ... * (365-n+1)) / (365^n))
#
# Now, the total amount of possible birthdays in a year given a random person is 365^1
# For 2 (random) people, it's 365^2 - and so on
#
# To find the total amount of birthdays with no duplicates, we calculate
# 365^n. You can view this as a 365 x 365 x n matrix. If we were to remove the duplicates,
# i.e. the equivalent of removing the diagonal of n 365x365 matrices, we find
# 365 * 364 * ... * (365-n+1)
#
# This finds the probability of n people with no duplicate birthdays. We take
# 1 - that and get the probability there are duplicate birthdays.


def birthday_paradox(n):
    arr = np.zeros((n, 365))  # create 3d numpy arr of size N x 365
    birthdays = np.random.randint(
        0, 365, n
    )  # assume birthdays are uniformly distributed
    arr[np.arange(n), birthdays] = 1  # assign each person a random birthday
    birthday_counts = arr.sum(axis=0)  # sum across first dimension
    return (birthday_counts > 1).any()  # if any day has more than 1 bday, True


TRIALS = 10_000

experiments = [birthday_paradox(N) for _ in range(TRIALS)]
prob = np.mean(experiments)

print(
    f"Estimated probability of a shared birthday in a group of {N} people: {prob}"
)
