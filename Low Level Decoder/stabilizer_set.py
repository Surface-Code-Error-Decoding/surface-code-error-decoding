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


stabilizerX_d3 = [[0,1], [7,8], [1,2,4,5], [3,4,6,7]]

stabilizerZ_d3 = [[2,5], [3,6], [0,1,3,4], [4,5,7,8]]


stabilizerX_d5 = [[0,1], [2,3], [21,22], [23,24], 
                  [1,2,6,7], [3,4,8,9], [5,6,10,11], [7,8,12,13], 
                  [11,12,16,17], [13,14,18,19], [15,16,20,21], [17,18,22,23]]

stabilizerZ_d5 = [[4,9], [14,19], [5,10], [15,20], 
                  [0,1,5,6], [2,3,7,8], [6,7,11,12], [8,9,13,14], 
                  [10,11,15,16], [12,13,17,18], [16,17,21,22], [18,19,23,24]]


