''' prover.py '''

from mpmath import iv
from mpmath import mp
from graphviz import Digraph
import sympy as sym
import sys
from sympy.plotting.intervalmath import interval

import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'


class Tree_node:
    default_segment = iv.mpf([0.0, 0.0])
    # операторы. Каждый оператор - список
    # нулевой элемент списка - приоритет
    # первый - тип оператора 'VAR' для переменной, 'OO' - для унарного оператора, 'TO' - для бинарного
    # второй - лямбда-функция самой операции
    # t = (2, 'TO', lambda x, y: x + y)
    # t[0] == 2
    # t[1] == 'TO'
    # t[2] == lambda x, y: x + y
    OPERATORS = {'+': (2, 'TO', lambda x, y: x + y), '-': (2, 'TO', lambda x, y: x - y),
                 '_': (2, 'UNARY', lambda x: -x),
                 '*': (3, 'TO', lambda x, y: x * y), '/': (3, 'TO', lambda x, y: x / y),
                 '^': (4, 'TO', lambda x, y: x ** y), 'x': (1, 'VAR', lambda: Tree_node.default_segment),
                 'sin': (5, 'OO', lambda x: iv.sin(x)), 'cos': (5, 'OO', lambda x: iv.cos(x)),
                 'ctg': (5, 'OO', lambda x: iv.cot(x)), 'tg': (5, 'OO', lambda x: iv.tan(x)),
                 'log': (5, 'OO', lambda x: iv.log(x)), 'exp': (5, 'OO', lambda x: iv.exp(x))}

    def __init__(self, operand, left=None, right=None):
        # для красивой печати. 
        self.name = operand
        # если в формуле число, то значение в вершине - сегмент
        if (type(operand) == float):
            # тип - чтобы отличать сегмент, операцию с одним операндом и операцию с двумя операндами
            self.type = 'segment'
            # это значение в формуле. Для числа и 'x' мы точно знаем результат
            self.value = iv.mpf(operand)
            # метка, что значение уже посчитано и пересчитывать не нужно (это для большой формулы долго)
            self.is_value_computed = True
            return
        # если в формуле 'x', то возвращаем сегмент со входа
        if (Tree_node.OPERATORS[operand][1] == 'VAR'):
            self.type = 'segment'
            self.value = Tree_node.default_segment
            self.is_value_computed = True
            return
        if (Tree_node.OPERATORS[operand][1] == 'OO'):
            self.type = 'OO'
            # для операции мы храним саму лямбда-функцию операции
            
            self.operation = Tree_node.OPERATORS[operand][2]
            # операнд всего один. Это вершина дерева
            self.subnode = left
            self.is_value_computed = False
            return
        if (Tree_node.OPERATORS[operand][1] == 'TO'):
            self.type = 'TO'
            self.operation = Tree_node.OPERATORS[operand][2]
            # левый и правый операнды формулы тоже вершины дерева
            self.left = left
            self.right = right
            self.is_value_computed = False
            return
        if (Tree_node.OPERATORS[operand][1] == 'UNARY'):
            self.type = 'UNARY'
            self.name = '-'
            self.operation = Tree_node.OPERATORS[operand][2]
            self.subnode = left
            self.is_value_computed = False

    # получаем значение в вершине
    def get_value(self):
        # если уже посчитано, то выдаем значение сразу
        if (self.is_value_computed):
            return self.value
        # если нет - считаем значение в поддереве
        if (self.type == 'OO'):
            self.value = self.operation(self.subnode.get_value())
            self.is_value_computed = True
            return self.value
        if (self.type == 'TO'):
            self.value = self.operation(self.left.get_value(), self.right.get_value())
            self.is_value_computed = True
            return self.value
        if (self.type == 'UNARY'):
            self.value = self.operation(self.subnode.get_value())
            self.is_value_computed = True
            return self.value

    def recompute_value(self):
        if (self.name == 'x'):
            return Tree_node.default_segment
        if (self.type == 'OO'):
            self.value = self.operation(self.subnode.recompute_value())
            self.is_value_computed = True
            return self.value
        if (self.type == 'TO'):
            self.value = self.operation(self.left.recompute_value(), self.right.recompute_value())
            self.is_value_computed = True
            return self.value
        if (self.type == 'UNARY'):
            self.value = self.operation(self.subnode.get_value())
            self.is_value_computed = True
            return self.value
        return self.value

    def get_visual(self, graph, id):
        if (self.is_value_computed):
            #iv.dps = 1
           # mp.dps = 1
            graph.node(name=str(id), label='%s\n%s' % (self.name, self.value.__str__()))
        else:
            graph.node(name=str(id), label='%s' % (self.name))
        if (self.type == 'UNARY'):
            self.subnode.get_visual(graph, id * 2)
            graph.edge(str(id), str(id * 2))
        if (self.type == 'OO'):
            self.subnode.get_visual(graph, id * 2)
            graph.edge(str(id), str(id * 2))
        elif (self.type == 'TO'):
            self.left.get_visual(graph, id * 2 + 1)
            self.right.get_visual(graph, id * 2)
            graph.edge(str(id), str(id * 2))
            graph.edge(str(id), str(id * 2 + 1))

    # для печати. 
    def __str__(self):
        if (self.type == 'segment'):
            return self.value.__str__()
        elif (self.type == 'OO' or self.type == 'UNARY'):
            if (self.is_value_computed):
                return '{}::( {} = {} )'.format(self.name, self.subnode.__str__(), self.value.__str__())
            return '{}::( {} )'.format(self.name, self.subnode.__str__())
        elif (self.type == 'TO'):
            if (self.is_value_computed):
                return '{}::( {} , {} = {})'.format(self.name, self.left.__str__(), self.right.__str__(),
                                                    self.value.__str__())
            return '{}::( {} , {} )'.format(self.name, self.left.__str__(), self.right.__str__())


# переводит формулу-строку в список (не совсем, просто строит генератор) из элементов, запись до сих пор псевдо-инфиксная:
# 'sin(',...,'tg(' переводит в ['(','sin'], ...
# переменная, число, бинарные операторы, '(' и ')' - в себя же
def parse(formula_string):
    number = ''
    # костыль для пропуска лишних символов у операторов, запись которых больше 1 символа
   
    skip_counter = 0
    is_unary = True
    for i in range(0, len(formula_string)):
        # пропускаем лишнее
        if (skip_counter > 0):
            skip_counter -= 1
            continue
        # обрабатываем числа и скобки - они собираются посимвольно
        s = formula_string[i]
        if s in '1234567890.':
            number += s
            continue
            # если символ не цифра, то выдаём собранное число и начинаем собирать заново
        elif number:
            yield float(number)
            is_unary = False
            number = ''
        if s in ('(', ')'):
            if s == '(':
                is_unary = True
            yield s
            continue
        if is_unary and s == '-':
            is_unary = False
            yield '_'
            continue
        # костыль для быстродействия - все операции по длине не превышают 3
        s = formula_string[i:i + 3]
        for op in Tree_node.OPERATORS:
            if s.find(op) == 0:
                is_unary = False
                # чтобы была правильная инфиксная запись проще сразу поменять 'sin' и '(' местами
                if Tree_node.OPERATORS[op][1] == 'OO':
                    is_unary = True
                    yield ('(')
                    skip_counter += 1
                skip_counter += len(op) - 1
                yield op

                break
    # если в конце строки есть число, выдаём его
    if number:
        is_unary = False
        yield float(number)

    # переводим формулу в дерево


# на выходе функции хотим получить корень дерева
#  всё происходит почти так же, как в переводе в обратную польскую запись
def infix_to_tree(parsed_formula):
    # для хранения необработанных операций
    stack = []
    # для хранения вершин без родителя
    node_stack = []
    for token in parsed_formula:
        if token == '(':
            stack.append(token)
        elif token == ')':
            # по закрывающей скобке можно получить корень поддерева, в котором содержится всё, что в скобках
            # log(...) + cos((...)...)
            # ( log ... ) + ( cos (...) ...)
            k = stack.pop()
            while k != '(':
                if (Tree_node.OPERATORS[k][1] == 'OO' or Tree_node.OPERATORS[k][1] == 'UNARY'):
                    node_stack.append(Tree_node(k, node_stack.pop()))
                elif (Tree_node.OPERATORS[k][1] == 'TO'):
                    right = node_stack.pop()
                    left = node_stack.pop()
                    node_stack.append(Tree_node(k, left, right))
                k = stack.pop()
        elif token in Tree_node.OPERATORS:
            if Tree_node.OPERATORS[token][1] == 'VAR':
                node_stack.append(Tree_node(token))
            else:
                while stack and stack[-1] != '(' and Tree_node.OPERATORS[stack[-1]][1] == 'TO' and \
                        Tree_node.OPERATORS[token][0] <= Tree_node.OPERATORS[stack[-1]][0]:
                    op = stack.pop()
                    right = node_stack.pop()
                    left = node_stack.pop()
                    node_stack.append(Tree_node(op, left, right))
                stack.append(token)
        else:
            node_stack.append(Tree_node(token))

    while stack:
        k = stack.pop()
        if (Tree_node.OPERATORS[k][1] == 'OO' or Tree_node.OPERATORS[k][1] == 'UNARY'):
            node_stack.append(Tree_node(k, node_stack.pop()))
        elif (Tree_node.OPERATORS[k][1] == 'TO'):
            right = node_stack.pop()
            left = node_stack.pop()
            node_stack.append(Tree_node(k, left, right))

    return node_stack[0]


def subdiv(tree_root, c):
    Tree_node.default_segment = c
    result_segment = tree_root.recompute_value()
    if (not (result_segment in iv.mpf([0, '+inf']) and not (0 in result_segment))):
        Tree_node.default_segment = c.a
        r_s_l = tree_root.recompute_value()
        Tree_node.default_segment = c.b
        r_s_r = tree_root.recompute_value()
        if ((r_s_l in iv.mpf(['-inf', 0])) or (r_s_r in iv.mpf(['-inf', 0]))):
            return [], [c]
        else:
            p_s_l, n_s_l = subdiv(tree_root, iv.mpf([c.a, c.mid]))
            p_s_r, n_s_r = subdiv(tree_root, iv.mpf([c.mid, c.b]))
            res_p = []
            res_n = []
            for i in p_s_l:
                res_p.append(i)
            for i in p_s_r:
                res_p.append(i)
            for i in n_s_l:
                res_n.append(i)
            for i in n_s_r:
                res_n.append(i)
            return res_p, res_n
    else:
        return [c], []







# функция получения производной
def derivative(expr):
    expr = expr.replace('^', '**')
    X = sym.Symbol('x')
    parsed_expr = sym.parse_expr(expr)
    return str(parsed_expr.diff(X)).replace('**', '^')


# функция вычисления значения
def calc(expr, x):
    expr = expr.replace('^', '**')
    parsed_expr = sym.parse_expr(expr)
    result = parsed_expr.evalf(subs={'x': x})
    return result


def seg(def_seg_str):
    if def_seg_str.find('(') == 0:
        int_type = 'interval'
    else:
        int_type = 'section'
    def_seg_str = def_seg_str.replace('[', '').replace('(', '').replace(']', '').replace(')', '').replace(',',
                                                                                                          ' ').split()
    if (def_seg_str[0].find('/') != -1):
        x_0 = float(def_seg_str[0][:def_seg_str[0].find('/')]) / float(def_seg_str[0][def_seg_str[0].find('/') + 1:])
    else:
        x_0 = float(def_seg_str[0])
    if (def_seg_str[1].find('/') != -1):
        x_1 = float(def_seg_str[1][:def_seg_str[1].find('/')]) / float(def_seg_str[1][def_seg_str[1].find('/') + 1:])
    else:
        x_1 = float(def_seg_str[1])
    return x_0, x_1, int_type


def find_eps(base_expr, x0, left=True, base_eps=1):
    expr = base_expr.replace('^', '**')
    x = sym.var('x')
    f = sym.parse_expr(expr)
    d = 0
    df = f
    while df.subs(x, x0) == 0:
        d += 1
        df = df.diff(x)
    print(f'{d}-th derivative of f is: {df}')
    eps = base_eps
    if df.is_number:
        idf = lambda a: df
    elif df.is_symbol:
        idf = lambda a: iv.mpf(a)
    else:
        idf = sym.lambdify(x, df, iv)
    if left:
        while not idf(iv.mpf([x0, x0 + eps])) > 0:
            if idf(iv.mpf([x0, x0 + eps])) < 0:
                return None
            eps *= 0.5
    else:
        # если чётная производная отрицательная или
        # нечётная положительная,
        # то функция не является положительной в отрезке.
        # FALSE
        if d % 2 == 0:
            while not idf(iv.mpf([x0 - eps, x0])) > 0:
                if idf(iv.mpf([x0, x0 + eps])) < 0:
                    print(f'even ({d}) derivative is negative')
                    return None
                eps *= 0.5
        else:
            while not idf(iv.mpf([x0 - eps, x0])) < 0:
                if idf(iv.mpf([x0, x0 + eps])) > 0:
                    print(f'odd ({d}) derivative is negative')
                    return None
                eps *= 0.5
    return eps

   

    # f(x) = x
    # f(0) = 0
    # f'(0) = 1, f'(0.0024) = 1

    # (0,45) -> [0.0024,44.9976]

    # Повторить для g([0,eps/2])

    # Поскольку  g(0) > 0, то для достаточно маленького  eps g([0,eps])>0 по непрерывности


# На вход поступает строка формулы, границы ОТРЕЗКА и флаг рисовать ли PDF
def is_positive(formula, x_0, x_1, render=False):
   
    def_seg = iv.mpf([x_0, x_1])
    # Задание интервала по-умолчанию
    Tree_node.default_segment = def_seg
    # получаем генератор из формулы
    print(formula)
    t = parse(formula)
    # получаем дерево из генератора
    p = infix_to_tree(t)
    print(p)
    # вычисляем все значения для дерева
    c = p.get_value()
    if render:
        # создаём объект ориентированого графа
        d = Digraph('formula')
        # отрисовываем все вершины и грани графа
        p.get_visual(d, 1)
        # отрисовываем дерево в test.pdf
        d.render('test', view=False, format='pdf')
    # if(c in iv.mpf(['-inf',0]) or (0 in c)):
    #     return False,c
    # else:
    #     return True,[]
    # Разбиение для дерева на основе интервала по-умолчанию
    q, v = subdiv(p, Tree_node.default_segment)
    # print(q)
    # print(v)
    # вывод True/False
    if (v == []):
        return True, len(q)
    else:
        return False, v[0]


def is_non_negative(formula, seg_val):
    try:
        base_expr = formula

        x_0, x_1, int_type = seg(seg_val)
        sys.setrecursionlimit(15000)
        int_x_0 = x_0
        int_x_1 = x_1
        # Если задан интервал, переходим к отрезку
        if int_type == 'interval':
            base_eps = 1
            if x_1 - x_0 <= base_eps:
                base_eps = (x_1 - x_0) / 2
            left_eps = find_eps(base_expr, x_0, True, base_eps)
            right_eps = find_eps(base_expr, x_1, False, base_eps)
            print('Интервал:', x_0, x_1)
            print('eps:', left_eps, right_eps)
            if left_eps == None or right_eps == None:
                if int_type == 'interval':
                    print('f(x)=' + formula + '\n is not positive on', '(', int_x_0, int_x_1, ')')
                else:
                    print('f(x)=' + formula + '\n is not positive on', '[', int_x_0, int_x_1, ']')
                return False
            x_0 = x_0 + left_eps
            x_1 = x_1 - right_eps
            print('Отрезок:', x_0, x_1)
        # если функция отрицательна на х0, эпс, то выводить фолс
        f_pos, _ = is_positive(formula, x_0, x_1, True)
        if f_pos:
            if int_type == 'interval':
                print('f(x)=' + formula + '\n is positive on', '(', int_x_0, int_x_1, ')')
            else:
                print('f(x)=' + formula + '\n is positive on', '[', int_x_0, int_x_1, ']')
            return True
        else:
            if int_type == 'interval':
                print('f(x)=' + formula + '\n is not positive on', '(', int_x_0, int_x_1, ')')
            else:
                print('f(x)=' + formula + '\n is not positive on', '[', int_x_0, int_x_1, ']')
            return False
    except:
        print('a','b','c')
        print(sys.exc_info())
        print('=^.^= Error :) Smth reeeeeally bad happend. Sorry. =^.^=')
        return None
   

def run_tests():
    data = [
        ('x^20*(1-x^10)', '(0,1)', True),
        ('sin(x)-sin(sin(x))', '(0,1)', True),
        ('x-sin(x)', '(0,1)', True),
        ('sin(sin(x))-x', '(0,1)', False),
        ('sin(x)-x', '(0,1)', False),
        ('cos(x)-1+x^2/2+1/1000', '[-2,1]', True),
        ('x^2-1', '(-2,2)', True),
        ('x^2+x-1', '(1,2)', True),
        ('x^2+x-1', '(-3,1)', True),
        ('(x+1)*(x^2-10/3*x+3)', '[1,3]', True),
        ('(x+1)*(x^2-10/3*x+3)', '(1,3)', True),
        ]

    for t in data:
        result = is_non_negative(t[0], t[1])
        answer = 'FAIL'
        if result == t[2]:
            answer = 'OK'
        print(t[0], t[1], 'expecting:', t[2], 'result:', result, answer)
