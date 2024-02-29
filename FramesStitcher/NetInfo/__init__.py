import sys
import os
dirPath = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(dirPath)
import NetworkInformationPyBind
import ItvCvUtilsPyBind
from pybind11_stubgen import ModuleStubsGenerator

def GeneratePyiFiles(moduleToParse):
    namePyi = f"{dirPath}/{moduleToParse.__name__}.pyi"
    if(os.path.exists(namePyi)):
        return
    print(moduleToParse.__name__)
    module = ModuleStubsGenerator(moduleToParse)
    module.parse()
    module.write_setup_py = False
    with open(namePyi, "w") as fp:
        fp.write("#\n# AUTOMATICALLY GENERATED FILE, DO NOT EDIT!\n#\n\n")
        fp.write("\n".join(module.to_lines()))

GeneratePyiFiles(NetworkInformationPyBind)
GeneratePyiFiles(ItvCvUtilsPyBind)