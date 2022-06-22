"""Tools for dealing with instances of Mapping"""

import os
import ast
import inspect

def tolerate_extra_args(f):
    """Returns an extra-argument tolerant version of f"""

    params = inspect.signature(f).parameters.keys()

    def tolerant_f(*args, **kwargs):
        selected_args = args[:len(params)]
        selected_kwargs = {}
        for p in params:
            try:
                selected_kwargs[p] = kwargs[p]
            except KeyError:
                pass
        return f(*selected_args, **selected_kwargs)

    return tolerant_f

def remedy(problem, remedy):
    """
    Function decorator to specify remedies for specific exceptions
    
    If problem arises in f(), remedy() will be called and f() will be re-run 
    """

    remedy = tolerate_extra_args(remedy)

    def robustify(f):
        def robust_f(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except problem:
                remedy(problem)
                return f(*args, **kwargs)

        return robust_f

    return robustify

def attempt(f, ex_type):
    """
    Function decorator to return None instead of raising exceptions
    """

    def relaxed_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ex_type:
            return
    return relaxed_f

environ = {name.lower(): 
             attempt(ast.literal_eval, (SyntaxError, ValueError))(val) 
           for name, val in os.environ.items()}

if __name__ == '__main__':
    def scream():
        return 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA!'

    nice_one = tolerate_extra_args(tolerate_extra_args)
    tolerant_scream = nice_one(scream, cucumber='green')
    print(tolerant_scream('hello', tomato='red'))