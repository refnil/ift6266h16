from blocks.serialization import load_parameter_values
from os.path import join
from os import listdir
from train import net_dvc
from theano import tensor, function
from blocks.model import Model
from datastream import get_dvc

folder = "checkpoints"
result_folder = "result"

def main(argv):
    name = argv[1]
    files = map(lambda p: join(folder, p), listdir(folder))

    file = next(filter(lambda n: name in n, files))
    print(file)

    p = load_parameter_values(file)

    net = net_dvc((128,128))

    x = tensor.tensor4('image_features')
    y_hat = net.apply(x)

    g = Model(y_hat)

    for k,v in p.items():
        p[k] = v.astype('float32')

    g.set_parameter_values(p)

    a,t,v = get_dvc((128,128),trainning=False, shortcut=False)
    run = function([x], y_hat)

    def run_test(data):
        res = []
        for i in  data.get_epoch_iterator():
            res.extend(run(i[0]))
        return res

    def max_index(l):
        if l[0] > l[1]:
            return 0
        else:
            return 1

    def write_kaggle(f, l):
        f.write("id,label\n")
        for i,e in enumerate(l,start=1):
            f.write(str(i)+","+str(e)+"\n")

    def kaggle(file, data):
        write_kaggle(file,map(max_index, run_test(data)))

    def accuracy(data):
        res = []
        true = []
        for i in data.get_epoch_iterator():
            res.extend(run(i[0]))
            true.extend(i[1])
        res = map(max_index, res)

        total = 0
        equal = 0
        for r,t in zip(res,true):
            total += 1
            equal += 1 if r == t else 0

        return equal/total

    print("Training accuracy: ", accuracy(a))
    print("Test accuracy: ", accuracy(v))
    kaggle_file = join(result_folder, name+".kaggle")
    print(kaggle_file)
    with open(kaggle,'w') as f:
            kaggle(f, t)


if __name__=="__main__":
    import sys
    main(sys.argv)
