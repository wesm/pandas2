# This file is a part of pandas. See LICENSE for details about reuse and
# copyright holders

import pytest

import codegen.types as ct

ValueType = ct.Typename('Value')


def test_vector():
    vec = ct.Vector('foo', ct.SharedPointer)

    assert vec.code() == 'foo'

    item = vec[1]
    assert isinstance(item, ct.SharedPointer)
    assert item.code() == 'foo[1]'


def test_pointers():
    sp = ct.SharedPointer('foo[1]')

    rp = sp.get()
    assert isinstance(rp, ct.RawPointer)
    assert rp.code() == 'foo[1].get()'

    ref = rp.deref()
    assert isinstance(ref, ct.Reference)
    assert ref.code() == '(*foo[1].get())'


def test_cast_constref():
    vec = ct.Vector('foo', ct.SharedPointer)

    type_ = ct.Typename('DoubleArray')

    val = type_.cast_constref(vec[1].get().deref())

    code, newval = val.assign('in1')
    assert isinstance(val, ct.ConstReference)
    assert code == (
        'const auto& in1 = '
        'static_cast<const DoubleArray&>((*foo[1].get()));')
    assert newval.code() == 'in1'


def test_assignment_auto():
    ref = ct.ConstReference('foo')
    code, lvalue = ref.assign('bar')

    assert code == 'const auto& bar = foo;'
    assert isinstance(lvalue, ct.ConstReference)
    assert lvalue.code() == 'bar'


def test_kind_check():
    ptr = ct.RawPointer('input[0]')

    check = ct.check_is_array(ptr)
    assert check.startswith(
        'PANDAS_DCHECK_EQ((input[0])->kind(), ValueKind::ARRAY)')


def test_template_function():
    type_ = ct.Typename('DoubleArray')

    tf = ct.TemplateFunction('MyTemplate')

    func = tf[type_, type_]

    assert isinstance(func, ct.Function)
    assert func.code() == 'MyTemplate<DoubleArray, DoubleArray>'
