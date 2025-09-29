# Lazy workaround to dynamically load classes from strings
# Recommended to use decorators or registries for larger projects

import importlib

def load(class_string, **kwargs):
    """
    Dynamically loads a class from a string.
    Format: "module_path.ClassName" or just "ClassName" for standard packages
    
    Args:
        class_string: String representation of the class
        **kwargs: Parameters to pass to the constructor
        
    Returns:
        Instantiated object
    """
    
    if '.' not in class_string:
        raise ValueError('class_string must be in the format "module_path.ClassName"')

    module_path, class_name = class_string.rsplit('.', 1)
    module = importlib.import_module(module_path)
    class_obj = getattr(module, class_name)
        
    if not kwargs: 
        return class_obj
    
    return class_obj(**kwargs)