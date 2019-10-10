import torch
import pymysql


def measure_time(func):
    def wrapper(*args, **kwargs):
        start = torch.cuda.Event(enable_timing = True)
        start.record()
        result = func(*args, **kwargs)
        end = torch.cuda.Event(enable_timing = True)
        end.record()
        torch.cuda.synchronize()
        training_time = start.elapsed_time(end)/1000
        return result, training_time
    return wrapper

def add_list(writer, data):
    for iteration, data in enumerate(data):
        writer.add_scalar(data, iteration)
    return writer

def add_graph(writer, model):
    dummy_input = torch.randn(1, 3, 224, 224).to(torch.device("cuda"))
    writer.add_graph(model, dummy_input)
    return writer

def save_env_on_filename(args):
    filename = ""
    for arg, value in (args.__dict__).items():
        if "path" in arg:
            continue
        if "gpu" in arg:
            continue
        filename += "{}_{}_".format(arg, value)
    return filename