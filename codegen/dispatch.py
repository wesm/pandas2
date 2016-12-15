# This file is a part of pandas. See LICENSE for details about reuse and
# copyright holders

from codegen.types import TypeCase, TypeSwitch, CallKernel


class DispatchRule:
    pass


class CartesianProduct(DispatchRule):

    def __init__(self, *arg_types):
        self.arg_types = arg_types

    def get_rule(self, kernel_name, args):
        return self._walk(kernel_name, args, 0, [])

    def _walk(self, kernel_name, args, level, selected_types):
        cases = []
        for type_ in self.arg_types[level]:
            branch_selected = selected_types + [type_]
            if level == len(self.arg_types) - 1:
                action = CallKernel(kernel_name, branch_selected, args)
            else:
                action = self._walk(kernel_name, args, level + 1,
                                    branch_selected)
            case = TypeCase(type_, action)
            cases.append(case)

        return TypeSwitch(args[level], cases)


class Kernel:

    def __init__(self, name, args, arg_types):
        self.name = name
        self.args = args
        self.arg_types = arg_types

    def format(self):
        pass


class Function:

    def __init__(self, name, kernel):
        pass
