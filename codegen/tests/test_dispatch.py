# This file is a part of pandas. See LICENSE for details about reuse and
# copyright holders

from codegen.dispatch import *

args = [ArrayArgument('values'), ArrayArgument('indices')]
dispatcher = CartesianProduct(ALL_TYPES, SIGNED_INTEGER_TYPES)

kernel = Kernel('take', args, dispatcher)

rule = dispatcher.get_rule('take', args)

print(rule.format())
