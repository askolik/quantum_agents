
import pickle


next_states = [
    [0, 4, 1, 0], # 0
    [0, 5, 2, 1], # 1
    [1, 6, 3, 2], # 2
    [2, 7, 3, 3], # 3
    [4, 8, 5, 0], # 4
    [4, 9, 6, 1], # 5
    [5, 10, 7, 2], # 6
    [6, 11, 7, 3], # 7
    [8, 12, 9, 4], # 8
    [8, 13, 10, 5], # 9
    [9, 14, 11, 6], # 10
    [10, 15, 11, 7], # 11
    [12, 12, 13, 8], # 12
    [12, 13, 14, 9], # 13
    [13, 14, 15, 10], # 14
    [14, 15, 15, 11] # 15
]


def get_opt_q_vals(gamma):
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

    return [q0, q1, q2, q3, q4, 0, q6, 0, q8, q9, q10, 0, 0, q13, q14, 0]


def get_all_transitions():
    state_space = list(range(16))
    action_space = list(range(4))

    transitions = []

    for state in state_space:
        for action in action_space:
            if state == 14 and action == 2:
                target = (state, action, 1, 15)
            else:
                target = (state, action, 0, next_states[state][action])

            transitions.append(target)

    return transitions


def get_optimal_transitions():
    # memory with 3 optimal routes to the goal stored as (state, action, reward, next_state),
    # not including duplicates
    return [
        # 1
        (0, 2, 0, 1),
        (1, 2, 0, 2),
        (2, 1, 0, 6),
        (6, 1, 0, 10),
        (10, 1, 0, 14),
        (14, 2, 1, 15),

        # 2
        (0, 1, 0, 4),
        (4, 1, 0, 8),
        (8, 2, 0, 9),
        (9, 1, 0, 13),
        (13, 2, 0, 14),

        # 3
        (9, 2, 0, 10),
        (10, 1, 0, 14)
    ]
