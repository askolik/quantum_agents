

def generate_all_bitstrings_of_size(n):
    enum_bitstrings = {}
    for i in range(n**2):
        binary = bin(i)[2:]
        template = ['0' for _ in range(4)]
        template[-len(binary):] = binary
        enum_bitstrings[i] = [int(x) for x in template]

    return enum_bitstrings


def get_frozen_lake_true_q_vals(state, gamma):
    q0 = [gamma ** 6, gamma ** 5, gamma ** 5, gamma ** 6]
    q1 = [gamma ** 5, 0, gamma ** 4, gamma ** 5]
    q2 = [gamma ** 5, gamma ** 3, gamma ** 5, gamma ** 4]
    q3 = [gamma ** 4, 0, gamma ** 5, gamma ** 5]
    q4 = [gamma ** 5, gamma ** 4, 0, gamma ** 6]
    q6 = [0, gamma ** 2, 0, gamma ** 4]
    q8 = [gamma ** 4, 0, gamma ** 3, gamma ** 5]
    q9 = [gamma ** 4, gamma ** 2, gamma ** 2, 0]
    q10 = [gamma ** 3, gamma, 0, gamma ** 3]
    q13 = [0, gamma ** 2, gamma, gamma ** 3]
    q14 = [gamma ** 2, gamma, 1, gamma ** 2]

    true_q_vals = [q0, q1, q2, q3, q4, 0, q6, 0, q8, q9, q10, 0, 0, q13, q14, 0]

    # binary_q = []
    # import numpy as np
    # for q in true_q_vals:
    #     if q != 0:
    #         vec = np.zeros(4)
    #         vec[np.argmax(q)] = 1
    #         binary_q.append(list(vec))
    #     else:
    #         binary_q.append(0)
    #
    # true_q_vals = binary_q

    q_val = [0, 0, 0, 0] if true_q_vals[state] == 0 else true_q_vals[state]

    return q_val


def argmax_all_max(arr):
    if len(arr) == 0:
        return []

    all_ = [0]
    max_ = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > max_:
            all_ = [i]
            max_ = arr[i]
        elif arr[i] == max_:
            all_.append(i)
    return all_
