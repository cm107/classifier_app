Next, I would like you to identify which classes should be broken into submodules.
I typically do this for complex classes that can be divided into specific branches that each have a distinctive role.

The submodules are defined in the original class as follows:
self.submodule1 = SubmoduleClass1(self)
self.submodule2 = SubmoduleClass2(self)
self.submodule3 = SubmoduleClass3(self)

Each submodule can then access the original class reference as well as other submodule reference like this:
self._parent # original class reference
self._parent.other_submodule # other submodule reference

This class structure is based on the "Separation of Concerns" principle. As a class gets more complex, it becomes easier to break it up into multiple submodules.

If any of the classes that you mentioned above can be broken up into submodules, please give me a submodule class name list for each of those classes. I will ask you for details later, so please just give me the submodule class names for now.