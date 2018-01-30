% Simulation Least Squares and Regularized Least Squares


%--------------------------------------------------------------------------
% In this simulation we'll prove that if the order of noise to signal ratio 
% is equal to the one of the eigenvalues of the matrix
% Sigma= X_tr'*X_tr 
% than regularized least squares with lambda=sigma^2/norm(w_star)^2 
% (which is unknown in a real problem) is better than least squares.
%--------------------------------------------------------------------------


%% I) n>>d stn<<1
    clear,clc, close all

    rng(3)
    % Dataset
%     a=randi([-100 -1]);
%     b=randi([1 100]);
    n=10^3;
    d=10;
%     sigma=randi([10^2 10^3]);
    
    w_star=rand(d,1);  % Exact w
    STN=10^(-1);
    sigma=sqrt(STN)*norm(w_star);
    
    % Trainingset
%     X_tr=rand(n,d)*(b-a)-b;
    X_tr=rand(n,d);
    Y_tr=X_tr*w_star+sigma*randn(size(X_tr,1),1);
    % Test set
%     X_ts=rand(n,d)*(b-a)-b;
    X_ts=rand(n,d);
    Y_ts=X_ts*w_star+sigma*randn(size(X_ts,1),1);

    % ------------------------------------------------------------------------------------------
    % Least Squares
    w_LS=(X_tr'*X_tr)\X_tr'*Y_tr;

    pred_err_LS=norm(X_ts*w_LS-X_ts*w_star)^2;

    % ----------------------------------------------------------------------------------------------
    % Regularized least Squares 
    lambda_star=sigma^2/norm(w_star)^2;
    lambdas=linspace(0,10^2*lambda_star,10^4);
    lambdas=sort([lambdas,lambda_star]);
    
    pred_err_RLS=zeros(1,size(lambdas,2));
    for i=1:size(lambdas,2)
        lambda=lambdas(i);
        w_RLS=(X_tr'*X_tr+lambda*eye(d))\X_tr'*Y_tr;
        pred_err_RLS(i)=norm(X_ts*w_RLS-X_ts*w_star)^2;
    end
    
    
% Errors plot
figure(1)
loglog(lambdas,pred_err_RLS,'linewidth',4)
hold on
loglog(lambdas,pred_err_LS*ones(1,size(lambdas,2)),'linewidth',2)
hold on
i=find(lambdas==lambda_star);
loglog(lambdas(i),pred_err_RLS(i),'.b','Markersize',30)
legend('RLS','LS')
title('n>>d    STN=0.1')


%% II) n>>d stn>>1
    clear,clc

    rng(3)
    % Dataset
%     a=randi([-100 -1]);
%     b=randi([1 100]);
    n=10^3;
    d=10;
%     sigma=randi([10^2 10^3]);
    
    w_star=rand(d,1);  % Exact w
    STN=10^(2);
    sigma=sqrt(STN)*norm(w_star);
    
    % Trainingset
%     X_tr=rand(n,d)*(b-a)-b;
    X_tr=rand(n,d);
    Y_tr=X_tr*w_star+sigma*randn(size(X_tr,1),1);
    % Test set
%     X_ts=rand(n,d)*(b-a)-b;
    X_ts=rand(n,d);
    Y_ts=X_ts*w_star+sigma*randn(size(X_ts,1),1);

    % ------------------------------------------------------------------------------------------
    % Least Squares
    w_LS=(X_tr'*X_tr)\X_tr'*Y_tr;
    
    pred_err_LS=norm(X_ts*w_LS-X_ts*w_star)^2;
    
    % ----------------------------------------------------------------------------------------------
    % Regularized least Squares 
    lambda_star=sigma^2/norm(w_star)^2;
    lambdas=linspace(0,10^2*lambda_star,10^4);
    lambdas=sort([lambdas,lambda_star]);
    
    pred_err_RLS=zeros(1,size(lambdas,2));
    for i=1:size(lambdas,2)
        lambda=lambdas(i);
        w_RLS=(X_tr'*X_tr+lambda*eye(d))\X_tr'*Y_tr;
        pred_err_RLS(i)=norm(X_ts*w_RLS-X_ts*w_star)^2;
    end
    
    
% Errors plot
figure(2)
loglog(lambdas,pred_err_RLS,'linewidth',4)
hold on
loglog(lambdas,pred_err_LS*ones(1,size(lambdas,2)),'linewidth',2)
hold on
i=find(lambdas==lambda_star);
loglog(lambdas(i),pred_err_RLS(i),'.b','Markersize',30)
legend('RLS','LS')
title('n>>d    STN=10^2')


%% III) n<<d stn<<1
    clear,clc

    rng(2)
    % Dataset
    n=10;
    d=10^2;
    
    w_star=rand(d,1);  % Exact w
    STN=10^(-1);
    sigma=sqrt(STN)*norm(w_star);
    
    % Trainingset
    X_tr=rand(n,d);
    Y_tr=X_tr*w_star+sigma*randn(size(X_tr,1),1);
    % Test set
    X_ts=rand(n,d);
    Y_ts=X_ts*w_star+sigma*randn(size(X_ts,1),1);

    % ------------------------------------------------------------------------------------------
    % Least Squares
    w_LS=X_tr'*((X_tr*X_tr')\Y_tr);

    pred_err_LS=norm(X_ts*w_LS-X_ts*w_star)^2;

    % ----------------------------------------------------------------------------------------------
    % Regularized least Squares 
    lambda_star=sigma^2/norm(w_star)^2;
    lambdas=linspace(0,0.6*10^2*lambda_star,10^4);
    lambdas=sort([lambdas,lambda_star]);
    
    pred_err_RLS=zeros(1,size(lambdas,2));
    for i=1:size(lambdas,2)
        lambda=lambdas(i);
        w_RLS=(X_tr'*X_tr+lambda*eye(d))\X_tr'*Y_tr;
        pred_err_RLS(i)=norm(X_ts*w_RLS-X_ts*w_star)^2;
    end
    
    
% Errors plot
figure(3)
loglog(lambdas,pred_err_RLS,'linewidth',4)
hold on
loglog(lambdas,pred_err_LS*ones(1,size(lambdas,2)),'linewidth',2)
hold on
i=find(lambdas==lambda_star);
loglog(lambdas(i),pred_err_RLS(i),'.b','Markersize',30)
legend('RLS','LS')
title('n<<d    STN=0.1')


%% IV) n<<d stn>>1
    clear,clc

    rng(2)
    % Dataset
    n=10;
    d=10^2;
    
    w_star=rand(d,1);  % Exact w
    STN=10^(2);
    sigma=sqrt(STN)*norm(w_star);
    
    % Trainingset
    X_tr=rand(n,d);
    Y_tr=X_tr*w_star+sigma*randn(size(X_tr,1),1);
    % Test set
    X_ts=rand(n,d);
    Y_ts=X_ts*w_star+sigma*randn(size(X_ts,1),1);
    
    % ------------------------------------------------------------------------------------------
    % Least Squares
    w_LS=X_tr'*((X_tr*X_tr')\Y_tr);
    
    pred_err_LS=norm(X_ts*w_LS-X_ts*w_star)^2;
    
    % ----------------------------------------------------------------------------------------------
    % Regularized least Squares 
    lambda_star=sigma^2/norm(w_star)^2;
    lambdas=linspace(0,10*lambda_star,10^4);
    lambdas=sort([lambdas,lambda_star]);
    
    pred_err_RLS=zeros(1,size(lambdas,2));
    for i=1:size(lambdas,2)
        lambda=lambdas(i);
        w_RLS=(X_tr'*X_tr+lambda*eye(d))\X_tr'*Y_tr;
        pred_err_RLS(i)=norm(X_ts*w_RLS-X_ts*w_star)^2;
    end
    
    
% Errors plot
figure(4)
loglog(lambdas,pred_err_RLS,'linewidth',4)
hold on
loglog(lambdas,pred_err_LS*ones(1,size(lambdas,2)),'linewidth',2)
hold on
i=find(lambdas==lambda_star);
loglog(lambdas(i),pred_err_RLS(i),'.b','Markersize',30)
legend('RLS','LS')
title('n<<d    STN=10^2')

