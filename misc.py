import pystan
import pickle
from hashlib import md5

def StanModel_cache(
        model_code=None,
        model_file=None,
        model_name=None,
        verbose=True,
        **kwargs
) -> pystan.StanModel:
    """
    Use just as you would `stan`

    stolen from: https://pystan.readthedocs.io/en/latest/avoiding_recompilation.html#automatically-reusing-models
    """

    if model_code is None:
        if model_file is not None:
            with open(model_file, 'r') as f:
                model_code = f.read()
        else:
            raise AssertionError('model_file can not be None if model_code is None!')

    code_hash = md5(model_code.encode('ascii')).hexdigest()
    if model_name is None:
        cache_fn = 'cached-model-{}.pkl'.format(code_hash)
    else:
        cache_fn = 'cached-{}-{}.pkl'.format(model_name, code_hash)
    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except:
        sm = pystan.StanModel(model_code=model_code)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    else:
        if verbose:
            print("Using cached StanModel")
    return sm