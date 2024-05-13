%% CHEBFUN
% unzip('https://github.com/chebfun/chebfun/archive/master.zip')
% movefile('chebfun-master', 'chebfun'), addpath(fullfile(cd,'chebfun')), savepath

%%

clear
close all
clc

%% AC equation - reference solution
% https://www.chebfun.org/docs/guide/guide19.html

S = spinop('ac'); % u_t = 5e-3 * u_xx + u - u^3
S.domain = [0, 2 * pi];
S.tspan = [0, 4];
% S.tspan = 0:100:4; % to save solution at intermediate times

u = spin(S, 2048, 1e-3, 'plot', 'off');

figure
plot(u)
