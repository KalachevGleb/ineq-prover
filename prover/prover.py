# coding=utf-8

class CannotSolve(Exception):
    pass


def is_non_negative(expr, var, segm):
    """

    :param expr:  Формула, задающая функцию f от одной переменной
    :param var:   Имя переменной
    :param segm:  Отрезок [a,b], на котором требуется проверить неравенство

    :returns      Выполнено ли неравенство f(x)>0 на всём отрезке [a,b]
    """
    raise CannotSolve("Пока функция ничего не делает. Здесь будет алгоритм прорверки неравенства f(x) >= 0")
