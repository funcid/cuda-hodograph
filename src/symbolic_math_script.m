clear; clc
syms x
y = 2*(x + 3)^2 / (x^2+6*x+9);
disp('y =')
disp(y)
result = simplify(y);
disp('y simplified is')
disp(result)
t = mexFunction([double(result)]);