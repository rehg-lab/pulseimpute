from pydantic import BaseModel
import importlib
import os

class BaseConfig(BaseModel):
    type: str

    def config_fileorigin(self):
        return importlib.import_module(self.__module__).__file__


def importclass(path, classname, parameters):
    module_name = os.path.splitext(path)[0].replace("/", ".")
    module = __import__(module_name, fromlist=[''])
    return getattr(module, classname)(parameters)