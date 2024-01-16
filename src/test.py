import sys

try:
    from example import hello
except ImportError:
    sys.path.append("/home/dswook/python_workspace/enefit/")
    from example import hello


print(hello())
