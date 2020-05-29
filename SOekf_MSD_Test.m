%% Second Order EKF Test
% Ethan Rapp
% 5/25/20
% Applied to simple MSD system with unknown spring constant
close all; clear; clc;

%% Params
m    = 5;  % [kg],    mass
b    = 0.5;  % [N*s/m], damping coefficient
Vmag = 0.05; % [m^2/s], noise magnitude
Kval = 1.5;  % [N/m],   actual stiffness (to be estimated)

%% Equations
% States:  X = [x1; x2; x3] = [position; velocity; spring constant]
% Input:   U = F = Force input
% Meas:    Y = [x1; Fstiffness] (to make it nonlinear, otherwise Hxx would be 0 matrix)

syms t x1 x2 x3 U
X = [x1; x2; x3]; 


xdot = [x2; -x3*x1/m - b*x2/m - U/m; 0]; 
fx   = jacobian(xdot,X)*X; 
gx   = jacobian(xdot,U)*U;
hx   = [x1; x3*x1/m]; 

Lx = length(X);
Lf = length(fx);
Lh = length(hx);

Fx  = jacobian(fx,X);
Hx  = jacobian(hx,X);

for i = 1:Lx
    Fxx(:,:,i) = hessian(fx(i),X);
end
for i = 1:Lh
    Hxx(:,:,i) = hessian(hx(i),X);
end

%% Create equations for evaluation
% fxeqn = @(t,X,U) [X(2); -X(3)*X(1)/m - b*X(2)/m - U/m; 0];
% hxeqn = @(t,X,U) X(1)*X(2);


fxeqn  = matlabFunction(fx, 'Vars',[x1,x2,x3,U]);
Fxeqn  = matlabFunction(Fx, 'Vars',[x1,x2,x3,U]);
Fxxeqn = matlabFunction(Fxx,'Vars',[x1,x2,x3,U]);

hxeqn  = matlabFunction(hx, 'Vars',[x1,x2,x3,U]);
Hxeqn  = matlabFunction(Hx, 'Vars',[x1,x2,x3,U]);
Hxxeqn = matlabFunction(Hxx,'Vars',[x1,x2,x3,U]);

%% Simulation
Xsim0 = [0.5; 0; Kval];

dt = 0.001;
t  = 0:dt:2; Lt = length(t);
F  = 3*(sin(2*pi*4*t) + 1);

Xsim      = zeros(Lx,Lt); Xsim(:,1) = Xsim0;
Ysim      = zeros(Lh,Lt); 
Ysim(:,1) = hxeqn(Xsim0(1),Xsim0(2),Xsim0(3),F(1)); 
Ymeas     = Ysim;

% Simulate Base Model
tic
for i = 2:Lt
    Xsim(:,i) = Xsim(:,i-1) + fxeqn(Xsim(1,i-1),Xsim(2,i-1),Xsim(3,i-1),F(i))*dt;

    % Create Measurement Vector
    Ymeas(:,i) = hxeqn(Xsim(1,i-1),Xsim(2,i-1),Xsim(3,i-1),F(i));
    Ysim(:,i)  = Ymeas(:,i) + Vmag*randn(Lh,1);
end
toc

%% EKF Simulation
Xekf0 = [0.25; 0; 0.9];

Xekf = zeros(Lx,Lt); Xekf(:,1) = Xekf0;
Yekf = zeros(Lh,Lt); Yekf(:,1) = hxeqn(Xekf(1,1),Xekf(2,1),Xekf(3,1),F(1));

% Noise Magnitude
Rk = Vmag^2.*eye(Lh);

% Process and Covariance IC
P0    = 0.1*eye(Lx); Pplus = P0;
Gamma = eye(Lx);
Qk    = 0.1*eye(Lx);

% Simulate Base Model
tic
for i = 2:Lt
   
    % Propagate Model
    Xhat   = Xekf(:,i-1) + fxeqn(Xekf(1,i-1),Xekf(2,i-1),Xekf(3,i-1),F(i))*dt;
    Fxhat  = Fxeqn(Xekf(1,i-1),Xekf(2,i-1),Xekf(3,i-1),F(i));
    
    % Calculate Expected Output
    hxhat  = hxeqn(Xhat(1),Xhat(2),Xhat(3),F(i));
    Hxhat  = Hxeqn(Xhat(1),Xhat(2),Xhat(3),F(i));
    
    % Update Covariance
    Phat    = Pplus;
    Pdot    = Fxhat*Phat + Phat*Fxhat' + Gamma*Qk*Gamma'; % - Phat*Hxhat'*(Rk^-1)*Hxhat*Phat;
    Pminus  = Phat + Pdot*dt;

    % Measurement
    zhat  = hxhat;
    zmeas = Ysim(:,i);
    
    % Update
    KG        = Pminus*Hxhat'*(Hxhat*Pminus*Hxhat' + Rk)^(-1);
    Xekf(:,i) = Xhat + KG*(zmeas - zhat);
    Pplus     = Pminus - KG*Hxhat*Pminus;
    Yekf(:,i) = hxeqn(Xekf(1,i),Xekf(2,i),Xekf(3,i),F(i));
end
fprintf('\n FO EKF time to complete: ')
toc

%% SO EKF Simulation
Xso0 = Xekf0;

% Reinitialize
Xso   = zeros(Lx,Lt); Xso(:,1) = Xso0;
Yso   = zeros(Lh,Lt); Yso(:,1) = hxeqn(Xso(1,1),Xso(2,1),Xso(3,1),F(1));
Pplus = P0;

% Simulate Base Model
tic
for i = 2:Lt
   
    % Propagate Model
    fxdot  = fxeqn(Xso(1,i-1),Xso(2,i-1),Xso(3,i-1),F(i));
    Fxhat  = Fxeqn(Xso(1,i-1),Xso(2,i-1),Xso(3,i-1),F(i));
    Fxxhat = Fxxeqn(Xso(1,i-1),Xso(2,i-1),Xso(3,i-1),F(i));
    
    % Calculate second order term:
    % FxxP = (1/2)*sum(Ei*trace(Fxxi*Pminus)), 
    % where Ei is unit vector of length Lx from i = 1:Lx
    % and Pminus is covariance from last iteration
    FxxP = zeros(Lx,1);
    FPFP = zeros(Lx,Lx);
    for j = 1:Lx
        FxxP(j,1) = FxxP(j,1) + (1/2)*trace(squeeze(Fxxhat(:,:,j))*Pplus);
    end
    Xhat   = Xso(:,i-1) + (fxdot + FxxP)*dt;
    
    % FPFP is second order covariance update term
    for j = 1:Lx
        for k = 1:Lx
            Fj = squeeze(Fxxhat(:,:,j));
            Fk = squeeze(Fxxhat(:,:,k));
            FPFP(j,k) = FPFP(j,k) + 0.5*trace(Fj*Pplus*Fk*Pplus);
        end
    end
    
    % Calculate Expected Output
    hxhat  = hxeqn(Xhat(1),Xhat(2),Xhat(3),F(i));
    Hxhat  = Hxeqn(Xhat(1),Xhat(2),Xhat(3),F(i));
    Hxxhat = Hxxeqn(Xhat(1),Xhat(2),Xhat(3),F(i));
    
    % Update Covariance
    Phat    = Pplus;
    Pdot    = Fxhat*Phat + Phat*Fxhat' + FPFP + Gamma*Qk*Gamma';
    Pminus  = Phat + Pdot*dt;

    % Second Order Output Terms: 
    % HxxP for measurement update
    % HPHtP for KG update
    % HxxP  = (1/2)*sum(Ei*trace(Hxxi*Phat))
    % HPHP = (1/2)*sum(Ei*Ei'*trace(Hxxi*Phat*Hxxi*Phat))
    HxxP = zeros(Lh,1);
    HPHP = zeros(Lh,Lh);
    
    for j = 1:Lh
        HxxP(j) = HxxP(j) + (1/2)*trace(squeeze(Hxxhat(:,:,j))*Pminus);
    end
    
    for j = 1:Lh
        for k = 1:Lh
            Hj = squeeze(Hxxhat(:,:,j));
            Hk = squeeze(Hxxhat(:,:,k));
            HPHP(j,k) = HPHP(j,k) + 0.5*trace(Hj*Pminus*Hk*Pminus);
        end
    end
    
    % Measurement
    zhat  = hxhat;
    zmeas = Ysim(:,i);

    % Update
    KG       = Pminus*Hxhat'*(Hxhat*Pminus*Hxhat' + HPHP + Rk)^(-1);
    Xso(:,i) = Xhat + KG*(zmeas - (zhat + HxxP));
    Pplus    = Pminus - KG*Hxhat*Pminus;
    Yso(:,i) = hxeqn(Xso(1,i),Xso(2,i),Xso(3,i),F(i));
end
fprintf('\n SO EKF time to complete: ')
toc











%% Figures
figure;
if Lh > 1
    for i = 1:Lh
        subplot(Lh,1,i)
        plot(t,Ymeas(i,:),t,Yekf(i,:),'r--',t,Yso(i,:),':k')
        ylabel(['Y(',num2str(i),')']);
    end
else
    plot(t,Ymeas,t,Yekf,'r--',t,Yso,':k')
    ylabel(['Y(',num2str(i),')']);
end
xlabel('Time [s]'); 
legend('True','EKF','SO EKF')


% subplot(2,1,2)
% plot(t,Ysim(2,:),t,Ymeas(2,:),'--',t,Yekf(2,:),':k')

figure;
Ksim = Xsim(3,:);
Kekf = Xekf(3,:);
Kso  = Xso(3,:);
plot(t,Ksim,t,Kekf,'r--',t,Kso,':k')
xlabel('Time [s]'); ylabel('Stiffness [N/m]')
legend('True','EKF','SO EKF')
axis([0 t(end) 0 2])












