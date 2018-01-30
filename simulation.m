%% Simulation Least Squares and Regularized Least Squares


%--------------------------------------------------------------------------
% In this simulation we'll prove that if the order of noise to signal ratio 
% is equal to the one of the eigenvalues of the matrix
% Sigma= X_tr'*X_tr 
% than regularized least squares with lambda=sigma^2/norm(w_star)^2 
% (which is unknown in a real problem) is better than least squares.
%--------------------------------------------------------------------------



%% I) n>>d  STN big
clear,clc,close all
rng(1)
n_problemi=20;
for j=1:n_problemi
    
    % Dataset
    n=10^4;
    d=randi([10 20]);
    w_star=rand(d,1);  % Exact w
    STN=10^(4);
    sigma=sqrt(STN)*norm(w_star);

    % Training set
    X_tr=rand(n,d);
    % Test set
    X_ts=rand(n,d);

    n_prove=0.5*10^2;
    for i=1:n_prove 
        
        Y_tr=X_tr*w_star+sigma*randn(size(X_tr,1),1);
        Y_ts=X_ts*w_star+sigma*randn(size(X_ts,1),1);
        % ------------------------------------------------------------------------------------------
        % Least Squares
        w_LS=(X_tr'*X_tr)\X_tr'*Y_tr;

        % ----------------------------------------------------------------------------------------------
        % Regularized least Squares 
        lambda=sigma^2/norm(w_star)^2;
        w_RLS=(X_tr'*X_tr+lambda*eye(d))\X_tr'*Y_tr;

        err_LS(i)=norm(X_ts*w_LS-X_ts*w_star)^2;
        err_RLS(i)=norm(X_ts*w_RLS-X_ts*w_star)^2;

    end

    pred_err_LS(j)=mean(err_LS);
    pred_err_RLS(j)=mean(err_RLS);


%     % Better prediction
%     if pred_err_RLS(j)<=pred_err_LS(j)
%         res_pred(j)=1;
%     else 
%         res_pred(j)=-1;
%     end
    
    disp(j)
end

% % Histogram
% figure(1)
% subplot(2,2,1)
% hist(res_pred)
% title('N>>D')

% Errors plot
figure(2)
subplot(2,2,1)
semilogy(pred_err_LS,'Linewidth',4)
hold on
semilogy(pred_err_RLS,'Linewidth',2)
legend('LS','RLS \lambda^*','Location','Southeast')
title('n>>D    STN=10^4')



%% II) n>>d  STN small
clear,clc
rng(1)
n_problemi=20;
for j=1:n_problemi
    
    % Dataset
    n=10^4;
    d=randi([10 20]);
    w_star=rand(d,1);  % Exact w
    STN=10;
    sigma=sqrt(STN)*norm(w_star);

    % Training set
    X_tr=rand(n,d);
    % Test set
    X_ts=rand(n,d);

    n_prove=0.5*10^2;
    for i=1:n_prove 
        
        Y_tr=X_tr*w_star+sigma*randn(size(X_tr,1),1);
        Y_ts=X_ts*w_star+sigma*randn(size(X_ts,1),1);
        % ------------------------------------------------------------------------------------------
        % Least Squares
        w_LS=(X_tr'*X_tr)\X_tr'*Y_tr;

        % ----------------------------------------------------------------------------------------------
        % Regularized least Squares 
        lambda=sigma^2/norm(w_star)^2;
        w_RLS=(X_tr'*X_tr+lambda*eye(d))\X_tr'*Y_tr;

        err_LS(i)=norm(X_ts*w_LS-X_ts*w_star)^2;
        err_RLS(i)=norm(X_ts*w_RLS-X_ts*w_star)^2;

    end

    pred_err_LS(j)=mean(err_LS);
    pred_err_RLS(j)=mean(err_RLS);


%     % Better prediction
%     if pred_err_RLS(j)<=pred_err_LS(j)
%         res_pred(j)=1;
%     else 
%         res_pred(j)=-1;
%     end
    
    disp(j)
end

% % Histogram
% figure(1)
% subplot(2,2,1)
% hist(res_pred)
% title('N>>D')

% Errors plot
figure(2)
subplot(2,2,2)
semilogy(pred_err_LS,'Linewidth',4)
hold on
semilogy(pred_err_RLS,'Linewidth',2)
legend('LS','RLS \lambda^*','Location','Southeast')
title('n>>D    STN=10')





%% III) n<<d  STN big
clear,clc
rng(1)
n_problemi=20;
for j=1:n_problemi
    
    % Dataset
    n=10;
    d=10^2;
    w_star=rand(d,1);  % Exact w
    STN=10^(4);
    sigma=sqrt(STN)*norm(w_star);

    % Training set
    X_tr=rand(n,d);
    % Test set
    X_ts=rand(n,d);

    n_prove=0.5*10^2;
    for i=1:n_prove   
        
        Y_tr=X_tr*w_star+sigma*randn(size(X_tr,1),1);
        Y_ts=X_ts*w_star+sigma*randn(size(X_ts,1),1);
        % ------------------------------------------------------------------------------------------
        % Least Squares
        w_LS=X_tr'*((X_tr*X_tr')\Y_tr);

        % ----------------------------------------------------------------------------------------------
        % Regularized least Squares 
        lambda=sigma^2/norm(w_star)^2;
        w_RLS=(X_tr'*X_tr+lambda*eye(d))\X_tr'*Y_tr;

        err_LS(i)=norm(X_ts*w_LS-X_ts*w_star)^2;
        err_RLS(i)=norm(X_ts*w_RLS-X_ts*w_star)^2;

    end

    pred_err_LS(j)=mean(err_LS);
    pred_err_RLS(j)=mean(err_RLS);


%     % Better prediction
%     if pred_err_RLS(j)<=pred_err_LS(j)
%         res_pred(j)=1;
%     else 
%         res_pred(j)=-1;
%     end
    
    disp(j)
end

% % Histogram
% figure(1)
% subplot(2,2,1)
% hist(res_pred)
% title('N>>D')

% Errors plot
figure(2)
subplot(2,2,3)
semilogy(pred_err_LS,'Linewidth',4)
hold on
semilogy(pred_err_RLS,'Linewidth',2)
legend('LS','RLS \lambda^*','Location','Southeast')
title('n<<D    STN=10^4')






%% IV) n<<d  STN small
clear,clc
rng(1)
n_problemi=20;
for j=1:n_problemi

    % Dataset
    n=10;
    d=10^2;
    w_star=rand(d,1);  % Exact w
    STN=10;
    sigma=sqrt(STN)*norm(w_star);

    % Training set
    X_tr=rand(n,d);
    % Test set
    X_ts=rand(n,d);

    n_prove=0.5*10^2;
    for i=1:n_prove 
        Y_tr=X_tr*w_star+sigma*randn(size(X_tr,1),1);
        Y_ts=X_ts*w_star+sigma*randn(size(X_ts,1),1);
        % ------------------------------------------------------------------------------------------
        % Least Squares
        w_LS=X_tr'*((X_tr*X_tr')\Y_tr);

        % ----------------------------------------------------------------------------------------------
        % Regularized least Squares 
        lambda=sigma^2/norm(w_star)^2;
        w_RLS=(X_tr'*X_tr+lambda*eye(d))\X_tr'*Y_tr;

        err_LS(i)=norm(X_ts*w_LS-X_ts*w_star)^2;
        err_RLS(i)=norm(X_ts*w_RLS-X_ts*w_star)^2;

    end

    pred_err_LS(j)=mean(err_LS);
    pred_err_RLS(j)=mean(err_RLS);


%     % Better prediction
%     if pred_err_RLS(j)<=pred_err_LS(j)
%         res_pred(j)=1;
%     else 
%         res_pred(j)=-1;
%     end
    
    disp(j)
end

% % Histogram
% figure(1)
% subplot(2,2,1)
% hist(res_pred)
% title('N>>D')

% Errors plot
figure(2)
subplot(2,2,4)
semilogy(pred_err_LS,'Linewidth',4)
hold on
semilogy(pred_err_RLS,'Linewidth',2)
legend('LS','RLS \lambda^*','Location','Southeast')
title('n<<D    STN=10')







