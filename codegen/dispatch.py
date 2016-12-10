# This file is a part of pandas. See LICENSE for details about reuse and
# copyright holders


class Argument:

    def __init__(self, name):
        self.name = name


class ArrayArgument(Argument):

    def __init__(self, name):
        Argument.__init__(self, name)


class ScalarArgument(Argument):

    def __init__(self, name):
        Argument.__init__(self, name)


class Type:

    def __init__(self, name):
        self.name = name
        self.array_type = '{0}Array'.format(self.name)
        self.scalar_type = '{0}Array'.format(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def type_id(self):
        return 'TypeId::{0}'.format(self.name.upper())

    def cast(self, arg):
        if isinstance(arg, ArrayArgument):
            cast_type = self.array_type
        elif isinstance(arg, ScalarArgument):
            cast_type = self.scalar_type

        return ('static_cast<const {0}&>(*{1}.get())'
                .format(cast_type, arg.name))


class TypeCase:

    def __init__(self, type_, action):
        self.type_ = type_
        self.action = action

    def format(self):
        return '''case {0}:
{1}; break;'''.format(self.type_.type_id(),
                      indent(self.action.format(), INDENT_SIZE))


class NumericType(Type):
    pass


INT8 = NumericType('Int8')
INT16 = NumericType('Int16')
INT32 = NumericType('Int32')
INT64 = NumericType('Int64')

UINT8 = NumericType('UInt8')
UINT16 = NumericType('UInt16')
UINT32 = NumericType('UInt32')
UINT64 = NumericType('UInt64')
FLOAT32 = NumericType('Float')
FLOAT64 = NumericType('Double')


SIGNED_INTEGER_TYPES = [INT8, INT16, INT32, INT64]
UNSIGNED_INTEGER_TYPES = [UINT8, UINT16, UINT32, UINT64]
INTEGER_TYPES = SIGNED_INTEGER_TYPES + UNSIGNED_INTEGER_TYPES
FLOATING_TYPES = [FLOAT32, FLOAT64]

ALL_TYPES = INTEGER_TYPES + FLOATING_TYPES


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


class Action:

    def format(self):
        raise NotImplementedError


class CallKernel(Action):

    def __init__(self, kernel_name, types, args):
        self.kernel_name = kernel_name
        self.types = types
        self.args = args

    def format(self):
        casted_args = [type_.cast(arg) for type_, arg in
                       zip(self.types, self.args)]
        return '{0}({1});'.format(self.kernel_name, ', '.join(casted_args))


class TypeSwitch(Action):

    def __init__(self, arg, cases):
        self.arg = arg
        self.cases = cases

    def format(self):
        formatted_cases = '\n'.join(case.format() for case in self.cases)
        return '''\
switch ({0}.type_id()) {{
{1}
}}'''.format(self.arg.name, indent(formatted_cases, INDENT_SIZE))


INDENT_SIZE = 2


def indent(text, spaces):
    block = ' ' * spaces
    return '\n'.join(block + x for x in text.split('\n'))


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
