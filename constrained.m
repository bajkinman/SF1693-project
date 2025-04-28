%% constrained problem
% let u be the deformation of a membrane that has to lie above
% the curve psi. Since the membrane satisfies the laplace equation
% without the constraint we get a constrained minimization problem
clear;
clc;

N = 100;
dx = 1/N;

A = zeros(N-1,N-1);
for i=1:(N-1)
    A(i,i) = 2*N;
end
for i=1:(N-2)
    A(i,i+1) = -N;
    A(i+1,i) = -N;
end

f = @(x) 0.5*x'*A*x-0; % L=0 in this case
fprime = @(x) A*x;
hessian = @(x) A;
psi = @(x) -1.25 .*(x-0.2).*(x-0.8); % the constraint
%% Gradient method with optimal step length, with penalty

function alpha = findalpha(f,zeta,d)
    temp = @(a) f(zeta + a*d);
    alpha = fminbnd(temp,0,10);
end

xs = linspace(0,1,N+1);
tempPsi = arrayfun(psi,xs(2:end-1))';
u = max(0,tempPsi);

R=1; % coefficient for the penalty
costumMax = @(x) max(0,x)^2;
costumMaxPrime = @(i,x) 2*(x-tempPsi(i))*(x < tempPsi(i));

% Minimize F
F = @(x) f(x) + R*sum(arrayfun(costumMax,x-tempPsi));
Fprime = @(x) A*x + R*sum(arrayfun(costumMaxPrime,(1:N-1)',x));

diff = 1;
for x = 1:20000 % we fix the number of iterations
    d = -Fprime(u);
    alpha = findalpha(F,u,d);
    step = alpha*d;
    diff = norm(step);
    u = u+step;
end

fig=gcf;
fig.Position(3:4)=[550,250];
plot(xs,psi(xs),'LineWidth',1.25);
hold on
plot(xs,[0;u;0],'LineWidth',2)
legend('\psi','u');
axis equal
