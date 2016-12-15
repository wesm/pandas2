# This file is a part of pandas. See LICENSE for details about reuse and
# copyright holders

from codegen.types import Action


class CallKernel(Action):

    def __init__(self, kernel_name, types, args):
        self.kernel_name = kernel_name
        self.types = types
        self.args = args

    def format(self):
        casted_args = [type_.cast(arg) for type_, arg in
                       zip(self.types, self.args)]
        return '{0}({1});'.format(self.kernel_name, ', '.join(casted_args))



# args = [ArrayArgument('values'), ArrayArgument('indices')]
# dispatcher = CartesianProduct(ALL_TYPES, SIGNED_INTEGER_TYPES)

# kernel = Kernel('take', args, dispatcher)

# rule = dispatcher.get_rule('take', args)

# print(rule.format())
