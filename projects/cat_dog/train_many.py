import train
import config
import sys

if __name__=="__main__":
    jobid = sys.argv[0]
    arrayid = int(sys.argv[1])

    params = config.parameters_list[arrayid]
    params.append("-j " + jobid)

    params = " ".join(params).split()

    train.main(params)
