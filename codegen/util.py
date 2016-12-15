# This file is a part of pandas. See LICENSE for details about reuse and
# copyright holders


def indent(text, spaces):
    block = ' ' * spaces
    return '\n'.join(block + x for x in text.split('\n'))
