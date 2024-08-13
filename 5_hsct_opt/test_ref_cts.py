# test ref counts
import sys


class MyClass:
    pass


# myvar = "abdacadabda"
myvar = MyClass()
refct1 = sys.getrefcount(myvar)
print(f"refct1 = {refct1}")

mylist = [myvar]
refct2 = sys.getrefcount(myvar)
print(f"refct2 = {refct2}")
