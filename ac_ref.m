%%

clear
close all
clc

%% Problem definition

N = 2048;
% N = 512;
x = linspace(0, 2 * pi, N)';
L = 2 * pi;
h = L / N;

tspan = [0, 4];

% Paper [2]
epsilon = 5e-2;
a = @(x, t) 1.05 + t .* sin(x);
phi = @(x, w, b) exp(-w.^2 .* abs(sin(pi * (x - b) / L)).^2);
u0 = phi(x, sqrt(10), 0.5) - phi(x, sqrt(10), 4.4);

% Paper [1] (coherent with the result in the RSNG repo)
% epsilon = 5e-3;
% a = @(x, t) 1 + 0 .* x .* t;
% u0 = tanh(2 * sin(x)) / 3 - exp(-23.5 * (x - pi / 2).^2) + ...
%     exp(-27 * (x - 4.2).^2) + exp(-38 * (x - 5.4).^2);

A = -2 * eye(N) + diag(ones(N-1, 1), 1) + diag(ones(N-1, 1), -1);
A = sparse(A);

rhs = @(t, u) epsilon * A * u / h^2 - a(x, t) .* (u - u.^3);

%% Solve AC

[t, u] = ode15s(rhs, tspan, u0);
% [t, u] = ode23s(rhs, tspan, u0);

%% Heatmap

figure
hm = heatmap(u'); % x: time, y: space
colormap('jet')
grid off
colorbar

%% 3D plot

% figure
% surf(u')
% colorbar

%% Save solution to file

writematrix([t, u], 'ac_ref.txt')
