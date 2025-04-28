%% Gradient method with optimal step length
clear;
clc;

F = @(x) 1+(x(1)/2)^2+x(2)^2;
fprime = @(x) [x(1)/2,2*x(2)];

function alpha = findalpha(f,zeta,d)
    temp = @(a) f(zeta + a*d);
    alpha = fminbnd(temp,0,10);
end

curr = [0.8,0.6];
diff = 1;
tol = 1e-5;

hist = curr;
while diff > tol
    d = -fprime(curr);
    
    alpha = findalpha(F,curr,d);
    next = curr + alpha*d;

    diff = norm(curr-next);
    curr = next;
    hist = [hist;curr];
end

coords = [];
for i=1:size(hist,1)
    coords(end+1,:) = [hist(i,:),F(hist(i,:))];
end

X = linspace(-1,1,100);
Y = linspace(-1,1,100);
Z = zeros(100,100);
for i = 1:100
    for j = 1:100
        Z(i,j) = F([X(j),Y(i)]);
    end
end

plot3(coords(:,1),coords(:,2),coords(:,3),'-o','Color','r','MarkerFaceColor','r');
hold on
mesh(X,Y,Z);
