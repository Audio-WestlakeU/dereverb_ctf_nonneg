function x = opti_nonneg_l2fit(A,b,x,lambda)
% The optimization problem
%              min 1/2 ||Ax-b||^2
%       subject to x >= 0
%    x: nx1 (initialization of) optimization variable
%    A: mxn matrix
%    b: mx1
%    lambda: nx1 dual variable


para.mu = 20;                % t updata
para.alpha = 0.05;           % line search
para.beta = 0.5;             % line search
para.epsilon_feas = 1e-5;    % convergence
para.epsilon = 1e-5;         % convergence

[~,n] = size(A);
onen = ones(n,1);

ITE = 20;
for ite=1:ITE
    
    eta = x'*lambda;             % f(x) = -x;
    t = para.mu*n/eta;
    
    %% Search Direction
    
    % Derevative and Hessian of objective function
    D_obj = A'*(A*x-b);          % Derevative
    H_obj = A'*A;                % Hessian
    
    % r_t
    r_dual = D_obj-lambda;       % r_dual
    r_cent = lambda.*x-onen/t;   % r_cent
    
    if sqrt(r_dual'*r_dual)<=para.epsilon_feas && eta<=para.epsilon
        break;
    end
    
    r_t = [r_dual;r_cent];       % r_t
    
    % combaination matrix
    CombMat = [H_obj -eye(n); diag(lambda) diag(x)];
    
    % Search direction
    Deltay = -CombMat\r_t;
    
    Deltax = Deltay(1:n);
    Deltalambda = Deltay(n+1:end);
    
    %% Line Search
    
    lambdaRat = -lambda./Deltalambda;
    if sum(Deltalambda<0)==0
        s = 0.99;
    else
        lambdaRat = lambdaRat(Deltalambda<0);
        s = 0.99*min(1,min(lambdaRat));
    end
    
    while 1
        x_plus = x+s*Deltax;
        if sum(x_plus<0)==0
            break;
        else
            s = para.beta*s;
        end
    end
    
    rt2 = sqrt(r_t'*r_t);
    while 1
        x_plus = x+s*Deltax;
        lambda_plus = lambda+s*Deltalambda;
        
        % Derevative of objective function at x_plus
        D_obj_plus = A'*(A*x_plus-b);
        
        % Combaination vector at x_plus
        r_t_plus = [D_obj_plus-lambda_plus;lambda_plus.*x_plus-onen/t];
        
        if sqrt(r_t_plus'*r_t_plus)<=(1-para.alpha*s)*rt2
            break;
        else
            s = para.beta*s;
        end
    end
    
    x = x_plus;
    lambda = lambda_plus;    
end







