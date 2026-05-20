#!/usr/bin/python3
"""
SVM - BNPS Simulator with Fully-Parallel Within-Membrane Semantics
Copied from bnps3.py and patched for SVM federated learning via BNPS.

Key fix over bnps3.py:
  - PARALLEL within-membrane semantics: every program in a membrane resets
    ALL variables to the global pre-step snapshot before it runs.
    This means the 800 individual weight-send rules:
        pr = { w1 -> 1|w1_sub1 };
        pr = { w1 -> 1|w1_sub2 }; ...
    each correctly see w1 at its snapshot value (not 0 from prior consumption).
  - Global consumed-variable zeroing is still applied AFTER all membranes
    finish, so gradient variables are correctly reset between steps.

Original BNPS features preserved:
  - Boolean operators: ∧ (AND), ∨ (OR), ¬ (NOT), ⊕ (XOR)
  - Unicode arrow → and ASCII -> as production/distribution separator
  - BNPS .pep file format: bnps = { ... }
  - Serial mode:   python3 svm.py check.pep -n <steps>
  - Parallel mode: python3 svm.py check.pep -p <steps>
"""

import collections
import re
from enum import IntEnum
import logging
import time
import math
import os

# ─────────────────────────────────────────────────────────────
# Operator Types  (ordered by precedence — lower int = lower precedence)
# ─────────────────────────────────────────────────────────────
class OperatorType(IntEnum):
    left_brace   = 1

    # Logical operators (lowest precedence)
    logical_or   = 2
    logical_xor  = 3
    logical_and  = 4
    logical_not  = 5

    # Comparison operators
    eq           = 6
    ne           = 7
    lt           = 8
    le           = 9
    gt           = 10
    ge           = 11

    # Arithmetic operators
    add          = 12
    subtract     = 13
    multiply     = 14
    divide       = 15
    power        = 16
    negate       = 17  # unary minus

    # Math functions
    sin = 18; sind = 19; asin = 20; asind = 21
    cos = 22; cosd = 23; acos = 24; acosd = 25
    tan = 26; tand = 27; atan = 28; atand = 29
    atan2 = 30; atan2d = 31
    cot = 32; cotd = 33; acot = 34; acotd = 35
    sqrt = 36; abs = 37; log = 38; log10 = 39; log2 = 40


dictOperatorTypes = {
    'OPERATOR_ADD':              OperatorType.add,
    'OPERATOR_SUBTRACT':         OperatorType.subtract,
    'OPERATOR_NEGATE':           OperatorType.negate,
    'OPERATOR_MULTIPLY':         OperatorType.multiply,
    'OPERATOR_DIVIDE':           OperatorType.divide,
    'OPERATOR_POWER':            OperatorType.power,
    'OPERATOR_EQUAL':            OperatorType.eq,
    'OPERATOR_NOT_EQUAL':        OperatorType.ne,
    'OPERATOR_LESS_THAN':        OperatorType.lt,
    'OPERATOR_LESS_EQUAL':       OperatorType.le,
    'OPERATOR_GREATER_THAN':     OperatorType.gt,
    'OPERATOR_GREATER_EQUAL':    OperatorType.ge,
    'OPERATOR_LOGICAL_NOT':      OperatorType.logical_not,
    'OPERATOR_LOGICAL_AND':      OperatorType.logical_and,
    'OPERATOR_LOGICAL_OR':       OperatorType.logical_or,
    'OPERATOR_LOGICAL_XOR':      OperatorType.logical_xor,
    'FUNCTION_SIN':   OperatorType.sin,   'FUNCTION_SIND':  OperatorType.sind,
    'FUNCTION_ASIN':  OperatorType.asin,  'FUNCTION_ASIND': OperatorType.asind,
    'FUNCTION_COS':   OperatorType.cos,   'FUNCTION_COSD':  OperatorType.cosd,
    'FUNCTION_ACOS':  OperatorType.acos,  'FUNCTION_ACOSD': OperatorType.acosd,
    'FUNCTION_TAN':   OperatorType.tan,   'FUNCTION_TAND':  OperatorType.tand,
    'FUNCTION_ATAN':  OperatorType.atan,  'FUNCTION_ATAND': OperatorType.atand,
    'FUNCTION_ATAN2': OperatorType.atan2, 'FUNCTION_ATAN2D':OperatorType.atan2d,
    'FUNCTION_COT':   OperatorType.cot,   'FUNCTION_COTD':  OperatorType.cotd,
    'FUNCTION_ACOT':  OperatorType.acot,  'FUNCTION_ACOTD': OperatorType.acotd,
    'FUNCTION_SQRT':  OperatorType.sqrt,  'FUNCTION_ABS':   OperatorType.abs,
    'FUNCTION_LOG':   OperatorType.log,   'FUNCTION_LOG10': OperatorType.log10,
    'FUNCTION_LOG2':  OperatorType.log2,
}

Token = collections.namedtuple('Token', ['type', 'value', 'line', 'column'])


# ─────────────────────────────────────────────────────────────
# Core Classes
# ─────────────────────────────────────────────────────────────

class NumericalPsystem():
    def __init__(self):
        self.H = []
        self.membranes = {}
        self.structure = None
        self.variables = []
        self.csvFile = None

    def runSimulationStep(self):
        """
        Run one SVM-BNPS simulation step (FULLY PARALLEL within-membrane):
          Phase 1 - Evaluate boolean conditions using global pre-step snapshot
          Phase 2 - Compute production values; each program resets to snapshot
          Phase 3 - Distribute values via repartition rules

        KEY CHANGE from bnps3.py:
          Within a membrane, EVERY program resets ALL membrane variables to
          the global snapshot before it runs. This means repeated rules like:
              pr = { w1 -> 1|w1_sub1 };
              pr = { w1 -> 1|w1_sub2 };
          both correctly read w1 from snapshot (not 0 from prior consumption).
          Global consumed-variable zeroing still runs after all membranes,
          ensuring gradient variables are reset between training steps.
        """
        snapshot = {v.name: v.value for v in self.variables}
        var_lookup = {v.name: v for v in self.variables}
        consumed_names = set()

        # Phase 1 & 2
        for membraneName in self.H:
            membrane = self.membranes[membraneName]
            if not membrane.programs:
                continue

            # Predicates use snapshot values (read-only)
            for program in membrane.programs:
                if program.predicate:
                    for item in program.predicate.items:
                        if isinstance(item, Pobject):
                            item.value = snapshot.get(item.name, item.value)

            membrane.chosenProgramNr = [
                i for i, prg in enumerate(membrane.programs)
                if prg.isActivated()
            ]

            if membrane.chosenProgramNr:
                try:
                    membrane.newValue = []
                    for prgNr in membrane.chosenProgramNr:
                        # FIX: reset ALL membrane vars to snapshot before EACH program
                        # (fully parallel — each program sees the original snapshot values,
                        #  not values consumed/zeroed by earlier programs in this membrane)
                        for v in membrane.variables:
                            v.value = snapshot.get(v.name, v.value)
                            v.wasConsumed = False
                        program = membrane.programs[prgNr]
                        val = program.prodFunction.evaluate()
                        # Track consumed variables for global zeroing after all membranes
                        for item in program.prodFunction.items:
                            if isinstance(item, Pobject) and item.wasConsumed:
                                consumed_names.add(item.name)
                        membrane.newValue.append(val)
                except RuntimeError as e:
                    logging.error(f"Production error in {membraneName}: {e}")
                    raise

        # Zero consumed vars globally after ALL membranes finish
        # (resets gradient variables like grad_w1_sub1 to 0 for next step)
        for name in consumed_names:
            if name in var_lookup:
                var_lookup[name].value = 0

        # Phase 3: Distribute production values
        for membraneName in self.H:
            membrane = self.membranes[membraneName]
            if not membrane.programs or not membrane.chosenProgramNr:
                continue
            for i, prgNr in enumerate(membrane.chosenProgramNr):
                program = membrane.programs[prgNr]
                program.distribFunction.distribute(membrane.newValue[i])


    def simulate(self, stepByStepConfirm=False, maxSteps=-1, maxTime=-1):
        currentStep = 1
        startTime = currentTime = time.time()
        finalTime = currentTime + maxTime if maxTime > 0 else float('inf')

        if self.csvFile:
            self._writeCsvHeader()

        while True:
            self.runSimulationStep()
            currentTime = time.time()

            if self.csvFile:
                self._writeCsvStep(currentStep)

            if stepByStepConfirm:
                input("Press ENTER to continue")

            if currentTime >= finalTime:
                logging.warning("Maximum time limit exceeded")
                break
            if 0 < maxSteps <= currentStep:
                logging.warning("Maximum step limit reached")
                break

            currentStep += 1

        logging.info(
            f"Simulation completed in {currentStep} steps, "
            f"{currentTime-startTime:.2f} seconds"
        )
        self.print()

    def _writeCsvHeader(self):
        self.csvFile.write("step," + ",".join(v.name for v in self.variables) + "\n")

    def _writeCsvStep(self, step):
        self.csvFile.write(f"{step}," + ",".join(str(v.value) for v in self.variables) + "\n")

    def print(self, indentSpaces=2, toString=False, withPrograms=False):
        result = "num_ps = {\n"
        for name in self.H:
            membrane = self.membranes[name]
            result += " " * indentSpaces + f"{name}:\n"
            result += membrane.print(indentSpaces * 2, True, withPrograms)
        result += "}\n"
        if toString:
            return result
        print(result)

    def openCsvFile(self):
        timestamp = time.strftime("%d-%m-%Y_%H-%M-%S")
        self.csvFile = open(f"bnps_{timestamp}.csv", "w")
        self.csvFile.write("BNPS csv output\n")


class MembraneStructure(list):
    def __init__(self):
        list.__init__(self)


class Membrane():
    def __init__(self, parentMembrane=None):
        self.variables = []
        self.programs = []
        self.chosenProgramNr = []
        self.newValue = []
        self.parent = parentMembrane
        self.children = {}

    def print(self, indentSpaces=2, toString=False, withPrograms=False):
        result = " " * indentSpaces + "var = {"
        for var in self.variables:
            result += " %s: %f, " % (var.name, var.value)
        result += "}\n"
        if withPrograms:
            for i, program in enumerate(self.programs):
                result += " " * indentSpaces + "pr_%d = { %s }\n" % (i, program.print(0, True))
        if toString:
            return result
        print(result)


class Program():
    def __init__(self):
        self.prodFunction = None
        self.distribFunction = None
        self.predicate = None

    def print(self, indentSpaces=2, toString=False):
        pred_str = ""
        if self.predicate is not None:
            pred_str = f"| {self.predicate.infixExpression} →"
        result = " " * indentSpaces + (
            f"{self.prodFunction.infixExpression} {pred_str} {self.distribFunction.expression}"
        )
        if toString:
            return result
        print(result)

    def isActivated(self):
        if self.predicate is None:
            return True
        try:
            return bool(self.predicate.evaluate())
        except Exception as e:
            logging.error(f"Error evaluating predicate: {e}")
            return False


class ProductionFunction():
    """Evaluates arithmetic production expression using postfix (RPN) evaluation."""
    def __init__(self):
        self.infixExpression = ""
        self.postfixStack = []
        self.items = []

    def evaluate(self):
        stack = []
        for item in self.items:
            if type(item) in (int, float):
                stack.append(item)
            elif type(item) == Pobject:
                stack.append(item.value)
                item.wasConsumed = True
                # DO NOT zero immediately - batch reset happens after ALL programs run
            elif item == OperatorType.add:
                stack.append(stack.pop() + stack.pop())
            elif item == OperatorType.multiply:
                stack.append(stack.pop() * stack.pop())
            elif item == OperatorType.subtract:
                b, a = stack.pop(), stack.pop()
                stack.append(a - b)
            elif item == OperatorType.negate:
                # FIX TC09: unary minus — only pops ONE value, not two
                stack.append(-stack.pop())
            elif item == OperatorType.divide:
                b, a = stack.pop(), stack.pop()
                stack.append(a / b)
            elif item == OperatorType.power:
                b, a = stack.pop(), stack.pop()
                stack.append(a ** b)
            elif item == OperatorType.eq:
                b, a = stack.pop(), stack.pop()
                stack.append(int(a == b))
            elif item == OperatorType.ne:
                b, a = stack.pop(), stack.pop()
                stack.append(int(a != b))
            elif item == OperatorType.lt:
                b, a = stack.pop(), stack.pop()
                stack.append(int(a < b))
            elif item == OperatorType.le:
                b, a = stack.pop(), stack.pop()
                stack.append(int(a <= b))
            elif item == OperatorType.gt:
                b, a = stack.pop(), stack.pop()
                stack.append(int(a > b))
            elif item == OperatorType.ge:
                b, a = stack.pop(), stack.pop()
                stack.append(int(a >= b))
            elif item == OperatorType.sin:
                stack.append(math.sin(stack.pop()))
            elif item == OperatorType.cos:
                stack.append(math.cos(stack.pop()))
            elif item == OperatorType.sqrt:
                stack.append(math.sqrt(stack.pop()))
            elif item == OperatorType.abs:
                stack.append(math.fabs(stack.pop()))
            elif item == OperatorType.log:
                stack.append(math.log(stack.pop()))
            elif item == OperatorType.log10:
                stack.append(math.log10(stack.pop()))
            elif item == OperatorType.log2:
                stack.append(math.log2(stack.pop()))

        if len(stack) != 1:
            raise RuntimeError(f'Production evaluation error. Stack: {stack}, Items: {self.items}')
        return stack[0]


class Predicate():
    """
    Boolean condition evaluator.
    Supports: ∧ AND, ∨ OR, ¬ NOT, ⊕ XOR, =, !=, <, <=, >, >=
    """
    def __init__(self):
        self.infixExpression = ""
        self.postfixStack = []
        self.items = []

    def evaluate(self):
        stack = []
        for item in self.items:
            if type(item) in (int, float):
                stack.append(item)
            elif type(item) == Pobject:
                stack.append(item.value)
            elif item == OperatorType.eq:
                b, a = stack.pop(), stack.pop()
                stack.append(int(a == b))
            elif item == OperatorType.ne:
                b, a = stack.pop(), stack.pop()
                stack.append(int(a != b))
            elif item == OperatorType.lt:
                b, a = stack.pop(), stack.pop()
                stack.append(int(a < b))
            elif item == OperatorType.le:
                b, a = stack.pop(), stack.pop()
                stack.append(int(a <= b))
            elif item == OperatorType.gt:
                b, a = stack.pop(), stack.pop()
                stack.append(int(a > b))
            elif item == OperatorType.ge:
                b, a = stack.pop(), stack.pop()
                stack.append(int(a >= b))
            elif item == OperatorType.add:
                stack.append(stack.pop() + stack.pop())
            elif item == OperatorType.subtract:
                b, a = stack.pop(), stack.pop()
                stack.append(a - b)
            elif item == OperatorType.multiply:
                stack.append(stack.pop() * stack.pop())
            elif item == OperatorType.negate:
                stack.append(-stack.pop())
            elif item == OperatorType.logical_not:
                stack.append(int(not bool(stack.pop())))
            elif item == OperatorType.logical_and:
                b, a = stack.pop(), stack.pop()
                stack.append(int(bool(a) and bool(b)))
            elif item == OperatorType.logical_or:
                b, a = stack.pop(), stack.pop()
                stack.append(int(bool(a) or bool(b)))
            elif item == OperatorType.logical_xor:
                b, a = stack.pop(), stack.pop()
                stack.append(int(bool(a) ^ bool(b)))

        if len(stack) != 1:
            raise RuntimeError(f'Predicate evaluation error. Stack: {stack}')
        return stack[0]


class DistributionFunction(list):
    def __init__(self):
        list.__init__(self)
        self.proportionTotal = 0
        self.expression = ""

    def distribute(self, newValue):
        for rule in self:
            rule.variable.value += (rule.proportion / self.proportionTotal) * newValue


class DistributionRule():
    def __init__(self):
        self.proportion = 0
        self.variable = None


class Pobject():
    def __init__(self, name='', value=0):
        self.name = name
        self.value = value
        self.wasConsumed = False


# ─────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────

def tokenize(code):
    token_specification = [
        ('OPERATOR_LOGICAL_XOR',     r'⊕'),
        ('OPERATOR_LOGICAL_AND',     r'∧|\band\b'),
        ('OPERATOR_LOGICAL_OR',      r'∨|\bor\b'),
        ('OPERATOR_LOGICAL_NOT',     r'¬|!'),
        ('PROD_DISTRIB_SEPARATOR',   r'→|->'),
        ('FUNCTION_ASIND',  r'asind'), ('FUNCTION_ASIN',  r'asin'),
        ('FUNCTION_SIND',   r'sind'),  ('FUNCTION_SIN',   r'sin'),
        ('FUNCTION_ACOSD',  r'acosd'),('FUNCTION_ACOS',  r'acos'),
        ('FUNCTION_COSD',   r'cosd'), ('FUNCTION_COS',   r'cos'),
        ('FUNCTION_ATAN2D', r'atan2d'),('FUNCTION_ATAN2', r'atan2'),
        ('FUNCTION_ATAND',  r'atand'),('FUNCTION_ATAN',  r'atan'),
        ('FUNCTION_TAND',   r'tand'), ('FUNCTION_TAN',   r'tan'),
        ('FUNCTION_ACOTD',  r'acotd'),('FUNCTION_ACOT',  r'acot'),
        ('FUNCTION_COTD',   r'cotd'), ('FUNCTION_COT',   r'cot'),
        ('FUNCTION_SQRT',   r'sqrt'), ('FUNCTION_ABS',   r'abs'),
        ('FUNCTION_LOG10',  r'log10'),('FUNCTION_LOG2',  r'log2'),
        ('FUNCTION_LOG',    r'log'),
        ('NUMBER_FLOAT',  r'\d+\.\d+'),
        ('NUMBER',        r'\d+'),
        ('OPERATOR_NOT_EQUAL',     r'!='),
        ('OPERATOR_EQUAL',         r'=='),
        ('OPERATOR_LESS_EQUAL',    r'<='),
        ('OPERATOR_GREATER_EQUAL', r'>='),
        ('OPERATOR_LESS_THAN',     r'<'),
        ('OPERATOR_GREATER_THAN',  r'>'),
        ('ASSIGN',        r'='),
        ('END',           r';'),
        ('ID',            r'[\w]+'),
        ('L_BRACE',       r'\('),
        ('R_BRACE',       r'\)'),
        ('L_CURLY_BRACE', r'{'),
        ('R_CURLY_BRACE', r'}'),
        ('L_BRACKET',     r'\['),
        ('R_BRACKET',     r'\]'),
        ('COLUMN',        r','),
        ('OPERATOR_ADD',      r'\+'),
        ('OPERATOR_SUBTRACT', r'\-'),
        ('OPERATOR_NEGATE',   r'\~'),
        ('OPERATOR_MULTIPLY', r'\*'),
        ('OPERATOR_DIVIDE',   r'\/'),
        ('OPERATOR_POWER',    r'\^'),
        ('DISTRIBUTION_SIGN', r'\|'),
        ('NEWLINE',  r'\n'),
        ('COMMENT',  r'#'),
        ('SKIP',     r'[ \t]+'),
        ('MISMATCH', r'.'),
    ]

    tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
    line_num = 1
    line_start = 0
    in_comment = False

    for mo in re.finditer(tok_regex, code, re.UNICODE):
        kind = mo.lastgroup
        value = mo.group(kind)

        if kind == 'COMMENT':
            in_comment = True
        elif kind == 'NEWLINE':
            line_start = mo.end()
            line_num += 1
            in_comment = False
        elif kind == 'SKIP':
            pass
        elif kind == 'MISMATCH' and not in_comment:
            raise RuntimeError('%r unexpected on line %d' % (value, line_num))
        else:
            if in_comment:
                continue
            yield Token(kind, value, line_num, mo.start() - line_start)


# ─────────────────────────────────────────────────────────────
# Postfix helper
# ─────────────────────────────────────────────────────────────

def processPostfixOperator(postfixStack, operator):
    outputList = []
    if len(postfixStack) > 0 and operator > postfixStack[-1]:
        postfixStack.append(operator)
    elif len(postfixStack) > 0 and operator <= postfixStack[-1]:
        outputList.append(postfixStack.pop())
        postfixStack, extra = processPostfixOperator(postfixStack, operator)
        outputList.extend(extra)
    else:
        postfixStack.append(operator)
    return postfixStack, outputList


# ─────────────────────────────────────────────────────────────
# FIX: Unary minus detection helper
# ─────────────────────────────────────────────────────────────

def _is_unary_position(items, stack):
    """
    Returns True if a minus sign at the current position should be
    treated as unary negation rather than binary subtraction.
    Unary if: nothing has been pushed to items yet, OR the last
    thing pushed was an operator (not a value/variable).
    """
    if not items and not stack:
        return True
    # Check last item pushed — if it's an operator type, next minus is unary
    if items:
        last = items[-1]
        if isinstance(last, OperatorType):
            return True
    # Stack top being left_brace means we're right after '(' — unary
    if stack and stack[-1] == OperatorType.left_brace:
        return True
    return False


# ─────────────────────────────────────────────────────────────
# Predicate parser
# ─────────────────────────────────────────────────────────────

def parsePredicate(tokens, index, stop_types):
    """
    Parse a boolean condition. Single = treated as equality (==).
    Handles unary minus for negative literals like e = -1.
    """
    pred = Predicate()
    infix = ""
    stack = []
    items = []

    while index < len(tokens):
        t = tokens[index]
        if t.type in stop_types:
            break

        infix += " " + t.value

        if t.type == 'NUMBER':
            items.append(int(t.value))
        elif t.type == 'NUMBER_FLOAT':
            items.append(float(t.value))
        elif t.type == 'ID':
            items.append(t.value)
        elif t.type == 'L_BRACE':
            stack.append(OperatorType.left_brace)
        elif t.type == 'R_BRACE':
            while stack and stack[-1] != OperatorType.left_brace:
                items.append(stack.pop())
            if stack:
                stack.pop()
        elif t.type == 'ASSIGN':
            op = OperatorType.eq
            stack, out = processPostfixOperator(stack, op)
            items.extend(out)
        elif t.type == 'OPERATOR_SUBTRACT':
            # FIX: detect unary minus in predicate (e.g. e = -1)
            if _is_unary_minus_predicate(tokens, index, items, stack):
                # Next token must be a number — consume it as negative literal
                if index + 1 < len(tokens) and tokens[index + 1].type in ('NUMBER', 'NUMBER_FLOAT'):
                    index += 1
                    nt = tokens[index]
                    infix += tokens[index].value
                    val = -int(nt.value) if nt.type == 'NUMBER' else -float(nt.value)
                    items.append(val)
                else:
                    # Treat as negate operator
                    stack, out = processPostfixOperator(stack, OperatorType.negate)
                    items.extend(out)
            else:
                op = OperatorType.subtract
                stack, out = processPostfixOperator(stack, op)
                items.extend(out)
        elif t.type in dictOperatorTypes:
            op = dictOperatorTypes[t.type]
            stack, out = processPostfixOperator(stack, op)
            items.extend(out)

        index += 1

    while stack:
        items.append(stack.pop())

    pred.items = items
    pred.infixExpression = infix.strip()
    return index, pred


def _is_unary_minus_predicate(tokens, index, items, stack):
    """
    In predicate context, minus is unary if:
    - nothing pushed to items yet, OR
    - last item was an operator, OR
    - last item was left_brace on stack
    - next token is a number (so -1, -2 etc.)
    """
    next_is_number = (
        index + 1 < len(tokens) and
        tokens[index + 1].type in ('NUMBER', 'NUMBER_FLOAT')
    )
    if not next_is_number:
        return False

    if not items:
        return True
    last = items[-1]
    if isinstance(last, OperatorType):
        return True
    if stack and stack[-1] == OperatorType.left_brace:
        return True
    # FIX: after a comparison/logical op still on shunting-yard stack,
    # the next minus must be unary (e.g. "e = -1")
    COMPARISON_OPS = {
        OperatorType.eq, OperatorType.ne,
        OperatorType.lt, OperatorType.le,
        OperatorType.gt, OperatorType.ge,
        OperatorType.logical_and, OperatorType.logical_or,
        OperatorType.logical_xor, OperatorType.logical_not,
    }
    if stack and stack[-1] in COMPARISON_OPS:
        return True
    return False


# ─────────────────────────────────────────────────────────────
# Production function parser — with unary minus fix
# ─────────────────────────────────────────────────────────────

def parseProductionFunction(tokens, index, stop_types):
    """
    Parse arithmetic production expression into postfix (RPN).

    FIX TC09: Leading minus (e.g. -2*a) is now correctly emitted as
    OperatorType.negate instead of subtract, so the evaluator never
    tries to pop two values when only one is on the stack.
    """
    pf = ProductionFunction()
    items = []
    stack = []
    infix = ""
    prev_was_value = False   # True after pushing a number/variable/closing brace

    while index < len(tokens):
        t = tokens[index]
        if t.type in stop_types:
            break

        infix += " " + t.value

        if t.type == 'NUMBER':
            items.append(int(t.value))
            prev_was_value = True
        elif t.type == 'NUMBER_FLOAT':
            items.append(float(t.value))
            prev_was_value = True
        elif t.type == 'ID':
            items.append(t.value)
            prev_was_value = True
        elif t.type == 'L_BRACE':
            stack.append(OperatorType.left_brace)
            prev_was_value = False
        elif t.type == 'R_BRACE':
            while stack and stack[-1] != OperatorType.left_brace:
                items.append(stack.pop())
            if stack:
                stack.pop()
            prev_was_value = True
        elif t.type == 'OPERATOR_SUBTRACT':
            # FIX: if minus appears where a value is expected → unary negate
            if not prev_was_value:
                op = OperatorType.negate
            else:
                op = OperatorType.subtract
            stack, out = processPostfixOperator(stack, op)
            items.extend(out)
            prev_was_value = False
        elif t.type in dictOperatorTypes:
            op = dictOperatorTypes[t.type]
            stack, out = processPostfixOperator(stack, op)
            items.extend(out)
            prev_was_value = False
        elif t.type in ('L_CURLY_BRACE', 'L_BRACKET'):
            prev_was_value = False
            index += 1
            continue
        else:
            # Unknown token in production — stop
            break

        index += 1

    # Flush remaining operators
    while stack:
        top = stack.pop()
        if top != OperatorType.left_brace:
            items.append(top)

    pf.items = items
    pf.infixExpression = infix.strip()
    return index, pf


# ─────────────────────────────────────────────────────────────
# Distribution function parser
# ─────────────────────────────────────────────────────────────

def parseDistributionFunction(tokens, index):
    """
    Parse repartition rules like: 2|c + 1|a + 1|b
    Each rule is: <coefficient>|<variable_name>
    Returns (index_after_closing_brace, DistributionFunction)
    """
    df = DistributionFunction()
    current_rule = None

    while index < len(tokens):
        t = tokens[index]

        if t.type == 'R_CURLY_BRACE':
            df.expression = df.expression.strip()
            return index, df

        elif t.type == 'NUMBER':
            current_rule = DistributionRule()
            current_rule.proportion = int(t.value)
            df.expression += " " + t.value

        elif t.type == 'DISTRIBUTION_SIGN':
            df.expression += "|"

        elif t.type == 'ID' and current_rule is not None:
            current_rule.variable = t.value   # string — resolved later
            df.proportionTotal += current_rule.proportion
            df.append(current_rule)
            df.expression += t.value
            current_rule = None

        elif t.type == 'OPERATOR_ADD':
            df.expression += " +"

        # END or semicolon also closes distribution in some formats
        elif t.type == 'END':
            df.expression = df.expression.strip()
            return index, df

        index += 1

    df.expression = df.expression.strip()
    return index, df


# ─────────────────────────────────────────────────────────────
# Token processor  (recursive, membrane-aware)
# ─────────────────────────────────────────────────────────────

def process_tokens(tokens, parent, index):
    """
    Recursively process token list and build the P system.

    FIX TC03/04/11/12: Nested membranes are now parsed correctly.
    When inside a Membrane block, numeric IDs that appear in system.H
    are parsed as child membrane definitions using the same recursive
    process_tokens call. This handles structures like:
        1 = { var = {...}; pr = {...}; var0 = (...); 2 = { ... }; }
    """
    result = parent
    prev_token = tokens[index]
    distribRule = None

    while index < len(tokens):
        token = tokens[index]

        # ── NumericalPsystem ──────────────────────────────────
        if type(parent) == NumericalPsystem:
            if token.type == 'ASSIGN':
                if prev_token.value == 'H':
                    index, result.H = process_tokens(tokens, list(), index + 1)
                elif prev_token.value in result.H:
                    index, result.membranes[prev_token.value] = process_tokens(
                        tokens, Membrane(), index + 1
                    )
                elif prev_token.value == 'structure':
                    index, result.structure = process_tokens(
                        tokens, MembraneStructure(), index + 1
                    )
                else:
                    raise RuntimeError(
                        "Unexpected '%s' on line %d" % (prev_token.value, prev_token.line)
                    )

        # ── MembraneStructure ─────────────────────────────────
        elif type(parent) == MembraneStructure:
            if token.type in ('ID', 'NUMBER', 'L_BRACKET', 'R_BRACKET'):
                parent.append(token)
            elif token.type == 'END':
                return index, result
            else:
                raise RuntimeError(
                    "Unexpected '%s' on line %d" % (token.value, token.line)
                )

        # ── Membrane ──────────────────────────────────────────
        elif type(parent) == Membrane:
            if token.type == 'ASSIGN':
                if prev_token.value == 'var':
                    index, variables = process_tokens(tokens, list(), index + 1)
                    for var in variables:
                        result.variables.append(Pobject(name=var))

                elif prev_token.value == 'pr':
                    index, program = process_tokens(tokens, Program(), index + 1)
                    result.programs.append(program)

                elif prev_token.value == 'var0':
                    index, variables = process_tokens(tokens, list(), index + 1)
                    for i, var in enumerate(variables):
                        result.variables[i].value = (
                            float(var) if isinstance(var, str) else var
                        )

                else:
                    # FIX TC03/04/11/12: child membrane definition inside parent
                    # e.g.  2 = { var = { d }; pr = {...}; var0 = (0); };
                    # prev_token.value is the child membrane ID (e.g. "2")
                    child_id = prev_token.value
                    # We need to reach up to the NumericalPsystem to register this membrane.
                    # We parse the child membrane here and return it; the caller must store it.
                    # But since process_tokens doesn't have a reference to NumericalPsystem
                    # at this level, we store it on the membrane as _pending_children so
                    # readInputFile can register it after parsing.
                    index, child_membrane = process_tokens(tokens, Membrane(), index + 1)
                    if not hasattr(result, '_pending_children'):
                        result._pending_children = {}
                    result._pending_children[child_id] = child_membrane

            elif token.type == 'R_CURLY_BRACE':
                return index + 1, result

        # ── Program ───────────────────────────────────────────
        elif type(parent) == Program:
            if token.type == 'L_CURLY_BRACE':
                index += 1

                # GNPS format: { [ condition ] production → distribution }
                if index < len(tokens) and tokens[index].type == 'L_BRACKET':
                    index += 1
                    index, result.predicate = parsePredicate(
                        tokens, index, {'R_BRACKET'}
                    )
                    if index < len(tokens) and tokens[index].type == 'R_BRACKET':
                        index += 1

                    # Production (using dedicated parser)
                    index, result.prodFunction = parseProductionFunction(
                        tokens, index,
                        {'PROD_DISTRIB_SEPARATOR', 'DISTRIBUTION_SIGN', 'R_CURLY_BRACE'}
                    )

                # BNPS format: { production | condition → distribution }
                else:
                    # Parse production until | or →
                    index, result.prodFunction = parseProductionFunction(
                        tokens, index,
                        {'DISTRIBUTION_SIGN', 'PROD_DISTRIB_SEPARATOR', 'R_CURLY_BRACE'}
                    )

                    # Check if | is condition separator or distribution separator
                    if index < len(tokens) and tokens[index].type == 'DISTRIBUTION_SIGN':
                        has_arrow = any(
                            tokens[j].type == 'PROD_DISTRIB_SEPARATOR'
                            for j in range(index + 1, len(tokens))
                            if tokens[j].type != 'R_CURLY_BRACE'
                        )
                        # Look ahead properly
                        has_arrow = False
                        for j in range(index + 1, len(tokens)):
                            if tokens[j].type == 'R_CURLY_BRACE':
                                break
                            if tokens[j].type == 'PROD_DISTRIB_SEPARATOR':
                                has_arrow = True
                                break

                        if has_arrow:
                            index += 1  # skip |
                            index, result.predicate = parsePredicate(
                                tokens, index, {'PROD_DISTRIB_SEPARATOR'}
                            )
                        # else: no condition, | is distribution — stay at current index

                # → separator
                if index < len(tokens) and tokens[index].type == 'PROD_DISTRIB_SEPARATOR':
                    index += 1
                    index, result.distribFunction = parseDistributionFunction(tokens, index)
                else:
                    raise RuntimeError("Expected '→' or '->' at index %d" % index)

                if index < len(tokens) and tokens[index].type == 'R_CURLY_BRACE':
                    return index + 1, result
                else:
                    raise RuntimeError("Expected '}' to close program block at index %d" % index)

        # ── List ──────────────────────────────────────────────
        elif type(parent) == list:
            if token.type == 'ID':
                result.append(token.value)
            elif token.type == 'NUMBER':
                if prev_token.type == 'OPERATOR_SUBTRACT':
                    result.append(-int(token.value))
                else:
                    result.append(token.value)
            elif token.type == 'NUMBER_FLOAT':
                val = (
                    -float(token.value)
                    if prev_token.type == 'OPERATOR_SUBTRACT'
                    else float(token.value)
                )
                result.append(val)

        # ── Top-level fallthrough ─────────────────────────────
        else:
            if token.type == 'ASSIGN':
                index, result = process_tokens(tokens, NumericalPsystem(), index + 1)

        if token.type == 'END':
            return index, result

        prev_token = token
        index += 1

    return index, result


# ─────────────────────────────────────────────────────────────
# File reader
# ─────────────────────────────────────────────────────────────

def readInputFile(filename):
    logging.info("Reading input file: %s" % filename)
    with open(filename, encoding='utf-8') as f:
        code = f.read()

    tokens = list(tokenize(code))
    index, system = process_tokens(tokens, None, 0)

    # FIX TC03/04/11/12: Register all pending child membranes discovered
    # during recursive membrane parsing into system.membranes and system.H
    def register_pending(membrane):
        if hasattr(membrane, '_pending_children'):
            for child_id, child_membrane in membrane._pending_children.items():
                if child_id not in system.membranes:
                    system.membranes[child_id] = child_membrane
                    if child_id not in system.H:
                        system.H.append(child_id)
                register_pending(child_membrane)

    for m_name in list(system.H):
        register_pending(system.membranes[m_name])

    # Build global variable list (all variables across all membranes)
    for mem_name in system.H:
        membrane = system.membranes[mem_name]
        for var in membrane.variables:
            if not any(v.name == var.name for v in system.variables):
                system.variables.append(var)

    # FIX TC05/07/14/15: Cross-reference string IDs → Pobject instances
    # This must search ALL membranes' variables for distribution targets,
    # not just the local membrane — enabling cross-membrane repartition.
    all_vars = system.variables  # flat list of all Pobjects across all membranes

    for var in all_vars:
        for mem_name in system.H:
            membrane = system.membranes[mem_name]
            for pr in membrane.programs:
                # Production function variable resolution
                for i, item in enumerate(pr.prodFunction.items):
                    if isinstance(item, str) and item == var.name:
                        pr.prodFunction.items[i] = var

                # Predicate variable resolution (separate Pobject, read-only)
                if pr.predicate is not None:
                    for i, item in enumerate(pr.predicate.items):
                        if isinstance(item, str) and item == var.name:
                            pr.predicate.items[i] = Pobject(
                                name=var.name, value=var.value
                            )

                # Distribution rule variable resolution (cross-membrane!)
                for rule in pr.distribFunction:
                    if isinstance(rule.variable, str) and rule.variable == var.name:
                        rule.variable = var

    # Build membrane parent/child tree from structure
    currentMembrane = None
    prev_tok = system.structure[0]
    for tok in system.structure[1:]:
        if tok.type in ('ID', 'NUMBER'):
            if prev_tok.type == 'L_BRACKET':
                if currentMembrane is None:
                    currentMembrane = system.membranes[tok.value]
                else:
                    currentMembrane.children[tok.value] = system.membranes[tok.value]
                    system.membranes[tok.value].parent = currentMembrane
                    currentMembrane = system.membranes[tok.value]
            elif prev_tok.type == 'R_BRACKET':
                if currentMembrane is not None:
                    currentMembrane = currentMembrane.parent
        prev_tok = tok

    return system


# ─────────────────────────────────────────────────────────────
# Extract system for CUDA input.txt
# ─────────────────────────────────────────────────────────────

def extractSystem(system):
    prodFunction = ""
    distFunction = ""
    predFunction = ""
    numberOfPrograms = 0
    numberOfVariables = [0]
    variables = []
    posProd = [0]
    posDist = [0]
    posPred = [0]
    numberOfMembranes = len(system.H)
    variableDictionary = {}
    membraneCounter = 0
    membraneVariableList = []
    progMembrane = []  # 0-indexed membrane index for each program

    for m_name in system.H:
        membraneCounter += 1
        variableCounter = 0
        membraneVariableList.append(m_name)
        membrane = system.membranes[m_name]
        numberOfPrograms += len(membrane.programs)
        numberOfVariables.append(len(membrane.variables))
        for var in membrane.variables:
            variableCounter += 1
            membraneVariableList.append(var.name)
            variables.append(var.value)
            variableDictionary[var.name] = '$_' + str(variableCounter) + '_' + str(membraneCounter)

    memIdx = 0
    for m_name in system.H:
        membrane = system.membranes[m_name]
        for program in membrane.programs:
            progMembrane.append(memIdx)
            newProd = "(" + program.prodFunction.infixExpression.replace(" ", "") + ")"
            newDist = program.distribFunction.expression.replace(" ", "")

            pfVars = sorted(
                [item.name for item in program.prodFunction.items if isinstance(item, Pobject)],
                key=len, reverse=True
            )
            for v in pfVars:
                newProd = newProd.replace(v, variableDictionary[v])

            dfVars = sorted(
                [rule.variable.name for rule in program.distribFunction
                 if isinstance(rule.variable, Pobject)],
                key=len, reverse=True
            )
            for v in dfVars:
                newDist = newDist.replace(v, variableDictionary.get(v, v))

            posProd.append(len(newProd) + len(prodFunction))
            posDist.append(len(newDist) + len(distFunction))
            prodFunction += newProd
            distFunction += newDist

            if program.predicate:
                newPred = program.predicate.infixExpression.replace(" ", "")
                predVars = sorted(
                    [item.name for item in program.predicate.items if isinstance(item, Pobject)],
                    key=len, reverse=True
                )
                for v in predVars:
                    newPred = newPred.replace(v, variableDictionary[v])
                # Encode Unicode operators as ASCII for CUDA kernel
                # bnps.cu expects: 'and', 'or', 'xor', '!' (not Unicode symbols)
                newPred = newPred.replace('∧', 'and')
                newPred = newPred.replace('∨', 'or')
                newPred = newPred.replace('¬', '!')
                newPred = newPred.replace('⊕', 'xor')
                posPred.append(len(newPred) + len(predFunction))
                predFunction += newPred
            else:
                posPred.append(len(predFunction) + 1)
                predFunction += "1"

        memIdx += 1

    for pos in range(len(numberOfVariables)):
        if pos != 0:
            numberOfVariables[pos] += numberOfVariables[pos - 1]

    return [numberOfPrograms, numberOfMembranes, prodFunction, posProd,
            distFunction, posDist, predFunction, posPred,
            numberOfVariables, variables, membraneVariableList, progMembrane]


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if '--debug' in sys.argv or '-v' in sys.argv:
        logLevel = logging.DEBUG
    elif '--error' in sys.argv or '-v0' in sys.argv:
        logLevel = logging.ERROR
    else:
        logLevel = logging.INFO

    try:
        import colorlog
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)-8s %(message)s %(reset)s",
            log_colors={'DEBUG':'cyan','INFO':'green','WARNING':'yellow','ERROR':'red'}
        )
        colorlog.basicConfig(stream=sys.stdout, level=logLevel)
        colorlog.root.handlers[0].setFormatter(formatter)
    except ImportError:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logLevel)

    if len(sys.argv) < 2:
        print("BNPS Simulator — Numerical P Systems with Boolean Conditions")
        print()
        print("Usage:")
        print("  python3 bnps.py <input.pep> -n <steps>    Serial execution")
        print("  python3 bnps.py <input.pep> -p <steps>    Parallel (CUDA) execution")
        sys.exit(1)

    if '-n' in sys.argv and '-p' in sys.argv:
        logging.error("Flags -n and -p are mutually exclusive.")
        sys.exit(1)

    if '-n' not in sys.argv and '-p' not in sys.argv:
        logging.error("Must specify either -n (serial) or -p (parallel).")
        sys.exit(1)

    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        logging.error(f"Input file not found: {input_file}")
        sys.exit(1)

    step_mode = '--step' in sys.argv

    try:
        system = readInputFile(input_file)
    except Exception as e:
        logging.error(f"Failed to parse input file: {e}")
        sys.exit(1)

    if "--csv" in sys.argv:
        system.openCsvFile()

    # ── Serial mode ───────────────────────────────────────────
    if '-n' in sys.argv:
        try:
            nrSteps = int(sys.argv[sys.argv.index('-n') + 1])
        except (ValueError, IndexError):
            logging.error("Expected a number after -n. Example: -n 5")
            sys.exit(1)
        system.simulate(stepByStepConfirm=step_mode, maxSteps=nrSteps)

    # ── Parallel mode ─────────────────────────────────────────
    elif '-p' in sys.argv:
        try:
            nrSteps = int(sys.argv[sys.argv.index('-p') + 1])
        except (ValueError, IndexError):
            logging.error("Expected a number after -p. Example: -p 5")
            sys.exit(1)

        parsed = extractSystem(system)
        output_file = "input.txt"

        try:
            with open(output_file, "w") as f:
                f.write(str(parsed[0]) + "\n")
                f.write(str(parsed[1]) + "\n")
                f.write(str(len(parsed[2])) + " " + str(parsed[2]) + "\n")
                f.write(str(len(parsed[3])) + " ")
                for i in parsed[3]: f.write(str(i) + " ")
                f.write("\n")
                f.write(str(len(parsed[4])) + " " + str(parsed[4]) + "\n")
                f.write(str(len(parsed[5])) + " ")
                for i in parsed[5]: f.write(str(i) + " ")
                f.write("\n")
                f.write(str(len(parsed[6])) + " " + str(parsed[6]) + "\n")
                f.write(str(len(parsed[7])) + " ")
                for i in parsed[7]: f.write(str(i) + " ")
                f.write("\n")
                f.write(str(len(parsed[8])) + " ")
                for i in parsed[8]: f.write(str(i) + " ")
                f.write("\n")
                f.write(str(len(parsed[9])) + " ")
                for i in parsed[9]: f.write(str(i) + " ")
                f.write("\n")
                f.write(str(nrSteps) + "\n")
                for i in parsed[10]: f.write(str(i) + " ")
                f.write("\n")
                # Write progMembrane: 0-indexed membrane index for each program
                f.write(str(len(parsed[11])) + " ")
                for i in parsed[11]: f.write(str(i) + " ")
                f.write("\n")
            logging.info(f"Generated {output_file} for CUDA kernel")
        except IOError as e:
            logging.error(f"Failed to write {output_file}: {e}")
            sys.exit(1)

        # Check CUDA availability
        cuda_available = False
        try:
            import subprocess as _sp
            _r = _sp.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if _r.returncode == 0 and _r.stdout.strip():
                cuda_available = True
        except Exception:
            cuda_available = False

        if cuda_available:
            logging.info("CUDA GPU detected. Running CUDA kernel...")
            compile_result = os.system("nvcc bnps_fast.cu -o bnps_exec -lm -Wno-deprecated-gpu-targets")
            if compile_result != 0:
                logging.error("CUDA compilation failed.")
                sys.exit(1)
            run_result = os.system("./bnps_exec input.txt")
            if run_result != 0:
                logging.error("CUDA execution failed.")
                sys.exit(1)

        else:
            # CPU multiprocessing fallback
            import multiprocessing
            import multiprocessing.pool

            logging.warning("No CUDA GPU detected. Falling back to CPU multiprocessing.")
            logging.info(f"Using {multiprocessing.cpu_count()} CPU cores.")

            def evaluate_membrane(args):
                mem_name, programs_data = args
                results = []
                for prog in programs_data:
                    results.append({
                        "active":        prog["pred_result"],
                        "prod":          prog["prod_value"],
                        "distrib":       prog["distrib"],
                        "distrib_total": prog["distrib_total"],
                    })
                return mem_name, results

            def run_parallel_step(system):
                snapshot = {v.name: v.value for v in system.variables}
                var_lookup = {v.name: v for v in system.variables}
                consumed_names = set()
                all_results = []

                for mem_name in system.H:
                    membrane = system.membranes[mem_name]

                    # Predicates use snapshot (read-only)
                    for program in membrane.programs:
                        if program.predicate:
                            for item in program.predicate.items:
                                if isinstance(item, Pobject):
                                    item.value = snapshot.get(item.name, item.value)

                    for program in membrane.programs:
                        # Reset each membrane variable to snapshot before EACH program
                        # so broadcast rules (w1->1|w1_sub1, w1->1|w1_sub2, ...)
                        # all see the same non-zero snapshot value.
                        for v in membrane.variables:
                            v.value = snapshot.get(v.name, v.value)
                            v.wasConsumed = False

                        pred_result = program.isActivated()
                        prod_val = 0.0
                        if pred_result:
                            try:
                                prod_val = program.prodFunction.evaluate()
                                for item in program.prodFunction.items:
                                    if isinstance(item, Pobject) and item.wasConsumed:
                                        consumed_names.add(item.name)
                                        # Do NOT zero here — zeroing happens globally
                                        # after ALL programs finish (line below).
                                        item.wasConsumed = False
                            except Exception as e:
                                logging.error(f'Production error in {mem_name}: {e}')

                        distrib = [
                            (rule.variable.name if isinstance(rule.variable, Pobject)
                             else rule.variable, rule.proportion)
                            for rule in program.distribFunction
                        ]
                        distrib_total = program.distribFunction.proportionTotal
                        all_results.append((pred_result, prod_val, distrib, distrib_total))

                for vname in consumed_names:
                    if vname in var_lookup:
                        var_lookup[vname].value = 0

                for pred_result, prod_val, distrib, total in all_results:
                    if not pred_result or total == 0:
                        continue
                    for varname, proportion in distrib:
                        target = var_lookup.get(varname)
                        if target is not None:
                            target.value += (proportion / total) * prod_val


            startTime = time.time()
            for step in range(nrSteps):
                run_parallel_step(system)
            elapsed = (time.time() - startTime) * 1000
            logging.info(f"Time taken: {elapsed:.6f} ms")

            print("")
            print("=== Final Results ===")
            system.print()

    if system.csvFile:
        logging.info("Wrote CSV: %s" % system.csvFile.name)
        system.csvFile.close()