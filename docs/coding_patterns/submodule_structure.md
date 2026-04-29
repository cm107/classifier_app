# Submodule Structure
Submodules are a good way to organize the logic of complex classes.

## When To Use Submodules
* If a class can be broken up into individual branches, where each branch has a distinctive role, it is highly recommended to create a submodule for each branch.
* Classes naturally become easier to break into submodules as they become more complex.

## Usage Concepts
* The parent class instance has access to all submodule class instance references via self.submoduleX
* Each submodule class instance has a parent reference, and can access other submodules through the parent reference via self._parent.otherSubmodule

## Structure
```python
class Submodule1:
    def __init__(self, parent: MainClass):
        self._parent = parent

        # Submodule parameters go here
        self.x = 1
        self.y = 2
        ...
    
    def method1(self): ...

    def method2(self): ...

    def referencing_parent_example(self):
        return self._parent.a + self._parent.b
    
    def referencing_other_submodule(self):
        sub2 = self._parent.submodule2
        return self.x + sub2.y
    
    ...

class MainClass:
    def __init__(self, ...):
        # Main class parameters go here
        self.a = 1
        self.b = 2
        ...

        # Submodules
        self.submodule1 = Submodule1(self)
        self.submodule2 = Submodule2(self)
        self.submodule3 = Submodule3(self)

```