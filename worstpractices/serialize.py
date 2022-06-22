from typing import Any, Sequence, Type
import dataclasses

def serialize(obj):
    if dataclasses.is_dataclass(obj):
        obj = dataclasses.asdict(obj)
    
    if type(obj) == dict:
        return {k: serialize(v) for k, v in obj.items()}
    elif isinstance(obj, Sequence):
        return [serialize(x) for x in obj]
    else:
        return str(obj)

def deserialize(obj_type: Type, obj: Any):
    try:
        origin = obj_type.__origin__
    except AttributeError:
        origin = None

    if origin:
        if origin == dict:
            k_type, v_type = obj_type.__args__
            return {deserialize(k_type, k): deserialize(v_type, v) for k, v in obj.items()}
        elif origin == list:
            x_type = obj_type.__args__[0]
            return [deserialize(x_type, x) for x in obj]
    else:
        if type(obj) == obj_type:
            return obj
        elif type(obj) == dict:
            return obj_type(**obj)
        else:
            return obj_type(obj)