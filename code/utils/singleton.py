def singleton(cls):
    """
    Defines a singleton instance

    Add @singleton to a class to use it as a singleton class.
    """
    instances = {}

    def getinstance(*args, **kw):
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]

    return getinstance
