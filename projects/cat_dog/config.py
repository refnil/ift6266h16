from itertools import product

def optional(l):
    return l+[""]

def lambda_generator(i=0):
    from random import uniform
    return str(uniform(0.01,0.2))

dropout = optional(["-d"])
l1 = ["--L1 " + lambda_generator()]
l2 = ["--L2 " + lambda_generator()]
update = optional(["-u rmsprop"])
data= optional(["-a"])

std_parameters = ["-t 27000" , "-e"]
variable_parameters = [dropout, l1, l2, update, data]

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
