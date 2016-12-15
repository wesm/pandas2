# This file is a part of pandas. See LICENSE for details about reuse and
# copyright holders

from codegen.util import indent


INDENT_SIZE = 2


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


class Action:

    def format(self):
        raise NotImplementedError


DCHECK = 'PANDAS_DCHECK({0})'
DCHECK_EQ = 'PANDAS_DCHECK_EQ({0}, {1})'
DCHECK_NE = 'PANDAS_DCHECK_NE({0}, {1})'
DCHECK_LT = 'PANDAS_DCHECK_LT({0}, {1})'
DCHECK_LE = 'PANDAS_DCHECK_LE({0}, {1})'
DCHECK_GT = 'PANDAS_DCHECK_GT({0}, {1})'
DCHECK_GE = 'PANDAS_DCHECK_GE({0}, {1})'


class Fragment:

    def __init__(self, code):
        self._code = code

    def code(self):
        return self._code

    def assign(self, name):
        code = '{0} {1} = {2};'.format(self._auto(), name, self.code())
        return code, type(self)(name)

    def invoke(self, method, *args):
        formatted_args = ', '.join([arg.code() for arg in args])
        return '({0}).{1}({2})'.format(self.code(), method, formatted_args)

    def _auto(self):
        return 'auto'


class Vector(Fragment):

    def __init__(self, code, value_type):
        self.value_type = value_type
        Fragment.__init__(self, code)

    def __getitem__(self, i):
        code = '{0}[{1}]'.format(self.code(), i)
        return self.value_type(code)


class Reference(Fragment):
    pass


class ConstReference(Reference):

    def _auto(self):
        return 'const auto&'


class Typename:

    def __init__(self, name):
        self.name = name

    def cast_constptr(self, val):
        code = 'static_cast<const {0}*>({1})'.format(self.name, val.code())
        return RawPointer(code)

    def cast_ptr(self, val):
        code = 'static_cast<{0}*>({1})'.format(self.name, val.code())
        return RawPointer(code)

    def cast_constref(self, val):
        code = 'static_cast<const {0}&>({1})'.format(self.name, val.code())
        return ConstReference(code)


class PointerLike(Fragment):

    def invoke(self, method, *args):
        formatted_args = ', '.join([arg.code() for arg in args])
        return '({0})->{1}({2})'.format(self.code(), method, formatted_args)

    def deref(self):
        new_code = '(*{0})'.format(self.code())
        return Reference(new_code)


class RawPointer(PointerLike):
    pass


class SharedPointer(Fragment):

    def get(self):
        new_code = '{0}.get()'.format(self.code())
        return RawPointer(new_code)


class Function(Fragment):

    def __call__(self, *args):
        pass


class TemplateFunction(Fragment):

    def __getitem__(self, types):
        formatted_types = ', '.join([t.name for t in types])
        code = '{0}<{1}>'.format(self.code(), formatted_types)
        return Function(code)


class DebugCheck:

    def _add_logging(self, check):
        return '{0} << __FILE__ << ":" << __LINE__'.format(check)


class ValueTypeCheck(DebugCheck):

    def __init__(self, type_):
        self.type_ = type_

    def format(self, argname):
        pass


class ValueKindCheck(DebugCheck):

    def __init__(self, kind):
        self.kind = 'ValueKind::{0}'.format(kind.upper())

    def format(self, fragment):
        check = DCHECK_EQ.format(fragment.invoke('kind'), self.kind)
        return self._add_logging(check)


def check_is_array(fragment):
    return ValueKindCheck('array').format(fragment)


def check_is_scalar(fragment):
    return ValueKindCheck('array').format(fragment)


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
