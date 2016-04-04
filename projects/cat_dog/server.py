from datastream import get_dvc
import argparse
from threading import Thread
from fuel.server import start_server as fuel_start_server

def tuple_parse(string):
    try:
        x, y = map(int, string.split(','))
        return x, y
    except:
        raise argparse.ArgumentTypeError("Need a width and height")

def start_server(i,stream):
    port = 5557 + i
    print("starting server " , port)
    fuel_start_server(stream, port)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Open a datastream for the cat and dog dataset')
    parser.add_argument('--image_size', type=tuple_parse, help='(width, height) for the resize')
    parser.add_argument('--batch_size', type=int, help='Set the number of exemple per batch')

    args = vars(parser.parse_args())
    args= {k:v for k,v in args.items() if v is not None}

    streams = get_dvc(**args)
    for t in enumerate(streams):
        Thread(target=start_server, args = t).start()

