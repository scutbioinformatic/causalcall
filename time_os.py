import psutil
from time import time as timestamp


def time_os(function, args=tuple(), kwargs={}):
    '''Return `real`, `sys` and `user` elapsed time, like UNIX's command `time`
    cross-platform (win/linux)
    '''
    start_time, start_resources = timestamp(), psutil.cpu_times()
    print(start_resources)
    function(*args, **kwargs)
    end_resources, end_time = psutil.cpu_times(), timestamp()
    print(end_resources)

    return {'real': end_time - start_time,
            'sys': end_resources.system - start_resources.system,
            'user': end_resources.user - start_resources.user}
