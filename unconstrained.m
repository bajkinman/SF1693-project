%% unconstrained minimization problem
% we solve the equation
% u'' + (1+x)u = exp(x), x \in [0,1]
% u(0) = u(1) = 0
% by formulating the corresponding variational formulation
% and approximating it with finite elements. The variational
% problem is then solved by instead solving an equivalent
% minimization problem
clear;
clc;
N = 100;
dx = 1/N;
xs = linspace(0,1,N+1);

% construct matrix A
A = zeros(N-1,N-1);
for i=1:(N-1)
    a = xs(i);
    b = xs(i+1);
    c = xs(i+2);
    A(i,i) = N^2*(2*dx+(c^4-a^4)/4+(1-2*a)/3*(b^3-a^3)+(a^2-2*a)/2*(b^2-a^2)...
    +(1-2*c)/3*(c^3-b^3)+(c^2-2*c)/2*(c^2-b^2)+(a^2+c^2)*dx);
end
for i=1:(N-2)
    a = xs(i+1);
    b = xs(i+2);
    A(i,i+1) = -N^2*(dx+(b^4-a^4)/4+(1-a-b)/3*(b^3-a^3)+(a*b-a-b)/2*(b^2-a^2)+a*b*dx);
    A(i+1,i) = A(i,i+1);
end
% construct vector L
L = zeros(N-1,1);
for i = 1:N-1
    L(i) = N*(exp(xs(i+2))+exp(xs(i))-2*exp(xs(i+1)));
end
% L(v) = int_0^1 e^x phi_i
% e^((-1 + i)/N) (-e^(1/N) (-1 + N) + N) + e^(i/N) (1 - 2 i + N - e^(1/N) (-2 i + N))

% function to minimize along with derivative and hessian
F = @(x) 0.5*x'*A*x-0 - L'*x;
Fprime = @(x) A*x-L;
hessian = @(x) A;

function alpha = findalpha(f,zeta,d)
    temp = @(a) f(zeta + a*d);
    alpha = fminbnd(temp,0,10);
end

%% generate plots

start = ones(N-1,1);
tol = 1e-5;

hold on;
figure(1);

plot(xs,[0;gradient_descent(F,Fprime,start,tol);0],"c");
plot(xs,[0;newton(Fprime,hessian,start,tol);0],"r--");
plot(xs,[0;quasi_newton(F,Fprime,hessian,start,tol);0],"m-.");
plot(xs,[0;conjugate_gradient(F,Fprime,start,tol);0],"b:");

legend({'Gradient method with optimal step length',"Newton's method", "Quasi-Newton's method", "Generalized conjugate gradient"},'Location','south')
legend('boxoff')

%% minimization algs

% Gradient method with optimal step length
function u = gradient_descent(F,Fprime,start,tol)
    diff = tol+1;
    u = start;
    while diff > tol
        d = -Fprime(u);
        alpha = findalpha(F,u,d);
        step = alpha*d;
        diff = norm(step);
        u = u+step;
    end
end

% Newton's method
function u = newton(Fprime,hessian,start,tol)
    diff = tol+1;
    u = start;
    while diff > tol
        d = -hessian(u)\Fprime(u);
        alpha = 1;
        step = alpha*d;
        diff = norm(step);
        u = u + step;
    end
end

% Quasi-Newton method
function u = quasi_newton(F,Fprime,hessian,start,tol)
    diff = tol+1;
    u = start;
    while diff > tol
        d = -hessian(u)\Fprime(u);
        alpha = findalpha(F,u,d);
        step = alpha*d;
        diff = norm(step);
        u = u + step;
    end
end

% Generalized conjugate gradient 
function u = conjugate_gradient(F,Fprime,start,tol)
    diff = tol+1;
    u = start;
    g_curr = Fprime(u);
    d_curr = -g_curr;
    while diff > tol
        alpha = findalpha(F,u,d_curr);
        next = u + alpha*d_curr;
    
        g_next = Fprime(next);
        beta = dot(g_next,g_next)/dot(g_curr,g_curr);
        d_next = -g_next+beta*d_curr; % every Mth step, set d_next = -g_next?
    
        diff = norm(u-next);
        u = next;
        g_curr = g_next;
        d_curr = d_next;
    end
end
