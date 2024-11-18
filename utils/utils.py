from pydantic import BaseModel
import importlib
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

class BaseConfig(BaseModel):
    type: str

    def config_fileorigin(self):
        return importlib.import_module(self.__module__).__file__
