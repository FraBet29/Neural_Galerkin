%%

clear
close all
clc

%% Solve AC

x = linspace(0, 2 * pi, 2048);
t = linspace(0, 12, 12000); % dt = 0.001

m = 0;
sol = pdepe(m, @AllenCahn, @AllenCahnInit, @AllenCahnBC, x, t);

%% Plot

u = sol(:, :, 1);

surf(x, t, u)
xlabel('x')
ylabel('t')
zlabel('u(x, t)')
% view([150 25])

%% Problem definition

function [c, f, s] = AllenCahn(x, t, u, dudx)

epsilon = 5e-2;
a = @(x, t) 1.05 + t .* sin(x);

c = 1; % coefficient of dudt
f = epsilon * dudx; % heat term: dx(epsilon * dudx)
s = a(x, t) * (u - u.^3); % source term

end

function u0 = AllenCahnInit(x)

phi = @(x, w, b) exp(-w.^2 .* abs(sin(pi * (x - b) / (2 * pi))).^2);
u0 = phi(x, sqrt(10), 0.5) - phi(x, sqrt(10), 4.4);

end

function [pl, ql, pr, qr] = AllenCahnBC(xl, ul, xr, ur, t)

pl = AllenCahnInit(0);
ql = 0;
pr = AllenCahnInit(2 * pi);
qr = 0; 

end
