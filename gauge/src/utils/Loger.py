import sys
import traceback
from functools import wraps

class CustomException(Exception):
    pass


def throws(ex=Exception, er=ValueError):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            try:
                print('>>> [INFO]: ***{', f.__name__, '}*** ==> START')
                return f(*args, **kwargs)
            except ex:
                print('>>> [ERROR]:', er.__name__, ' >>>')
                traceback.print_exc(file=sys.stdout)

            except Exception:
                print('>>> [ERROR]: Exception >>>')
                traceback.print_exc(file=sys.stdout)

            except er:
                print('>>> [ERROR]: Error >>>')
                traceback.print_exc(file=sys.stdout)

        return wrapped

    return decorator
