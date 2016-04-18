from itertools import product

dropout = ["-d"]
l1 = ["--L1"]
l2 = ["--L2"]

std_parameters = ["--finish 1"]
variable_parameters = [dropout, l1, l2]

def create_parameters_list(std, var):
    def extend(l):
        def extend_inside(e):
            return l+e
        return extend_inside

    var = map(extend([""]), var)
    var = map(list,product(*var))

    return list(map(extend(std), var))

parameters_list = create_parameters_list(std_parameters, variable_parameters)


if __name__=="__main__":
    print(len(parameters_list))
