from prover import *
import sympy as sym

# Список тестов для функции is_non_negative.
# Каждый тест -- тройка (формула f(x), отрезок [a,b], истинность утверждения: "f(x)>=0 при a <= x <= b")
# TODO: Необходимо пополнить этот список
tests = [
    ('sin(x)', [0, sym.pi], True),
    ('cos(x)', [0, sym.pi], False),
    ('x-sin(x)', [0, 1], True),
    ('cos(x)+sin(x)', [0, 2], True),
    ('cos(x)+sin(x)', [0, 6], False),
    ('sin(x)-sin(sin(x))', [0, 1], True)
]


def run_tests():
    correct = 0
    wrong = 0
    failed = 0

    for expr, segm, answer in tests:
        try:
            r = is_non_negative(expr, 'x', segm)
            if r != answer:
                print(f'Wrong answer for f(x) = {expr} in {segm}: {r}, correct is: {answer}')
                wrong += 1
            else:
                correct += 1
        except CannotSolve as e:
            print(f'Failed to check f(x) = {expr} in {segm}: {e.args[0]}')
            failed += 1

    print('\n=========== Summary ==============\n')
    print(f'correct:      {correct}')
    print(f'wrong answer: {wrong}')
    print(f'failed:       {failed}')


if __name__ == '__main__':
    run_tests()
