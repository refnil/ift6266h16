from itertools import product

def optional(l):
    return l+[""]

def lambda_generator(i):
    return str([0.01, 0.05, 0.1][i])
    from random import uniform
    return str(uniform(0.01,0.2))

dropout = optional(["-d"])
l1 = optional(["--L1 " + lambda_generator(i) for i in range(1, 3)])
l2 = optional(["--L2 " + lambda_generator(i) for i in range(1, 3)])
update = optional(["-u rmsprop"])

std_parameters = ["--finish 30", "-e"]
variable_parameters = [dropout, l1, l2, update]

def create_parameters_list(std, var):
    def extend(l):
        def extend_inside(e):
            return l+e
        return extend_inside

    var = map(list,product(*var))

    return list(map(extend(std), var))

parameters_list = create_parameters_list(std_parameters, variable_parameters)

if __name__=="__main__":
    print(len(parameters_list))
