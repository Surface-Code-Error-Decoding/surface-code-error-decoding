import itertools

def generate_stabilizer_set(stabilizer):

    combinations_list = []
    
    for r in range(1, len(stabilizer)+1):

        combinations = list(itertools.combinations(stabilizer, r))
        combinations_list += combinations

    stabilizer_set = []

    for comb in combinations_list:
        each = []
        for i in comb:
            each += i

        final = []
        st = list(set(each))
        for s in st:
            if (each.count(s)%2 == 1):
                final.append(s)

        final = sorted(final)

        stabilizer_set.append(final)

    for stab in stabilizer_set:
        if (sorted(stab) != stab):
            print("Something wrong")
            return 0

    print(len(stabilizer_set))
    return stabilizer_set