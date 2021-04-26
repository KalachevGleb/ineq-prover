from prover import *
import sympy as sym

# Список тестов для функции is_non_negative.
# Каждый тест -- тройка (формула f(x), отрезок [a,b], истинность утверждения: "f(x)>=0 при a <= x <= b")
# TODO: Необходимо пополнить этот список
tests = [
    ('x^20*(1-x^10)', '(0,1)', True),
    ('sin(x)-sin(sin(x))', '(0,1)', True),
    ('x-sin(x)', '(0,1)', True),
    ('sin(sin(x))-x', '(0,1)', False),
    ('sin(x)-x', '(0,1)', False),
    ('cos(x)-1+x^2/2+1/1000', '[-2,1]', True),
    ('x^2-1', '(-2,2)', False),
    ('x^2+x-1', '(1,2)', True),
    ('x^2+x-1', '(-3,1)', False),
    ('(x+1)*(x^2-10/3*x+3)', '[1,3]', True),
    ('(x+1)*(x^2-10/3*x+3)', '(1,3)', True),
]


def run_tests():
    correct = 0
    wrong = 0
    failed = 0

    for expr, segm, answer in tests:
        try:
            r = is_non_negative(expr, segm)
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
