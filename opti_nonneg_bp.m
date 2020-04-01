function [x,obj] = opti_nonneg_bp(A,b,x,delta,lambda)
% basis pursuit optimization problem
%              min  |x|_1
%       subject to ||Ax-b||^2<=delta, x >= 0
%    x: nx1 (initialization of) optimization variable
%    A: mxn matrix
%    b: mx1 
%    delta: bound of l2 fit cost
%    lambda: (n+1)x1 dual variable

para.mu = 20;              % t updata
para.alpha = 0.05;         % line search
para.beta = 0.5;           % line search
para.epsilon_feas = 1e-5;  %
para.epsilon = 1e-5;       %

[~,n] = size(A);

onen = ones(n+1,1);

% Derevative of objective function
D_obj = ones(n,1);
% Hessian of l2 constraint
H_conl2 = A'*A;

ITE = 15;
for ite=1:ITE    
    % constraint  
    con = [-x;norm(A*x-b,2)^2-delta];
    
    eta = -con'*lambda;
    t = para.mu*(n+1)/eta;
    
    %% Search Direction
    
    % Derevative and Hessian of constraint function
    D_conl2 = A'*(A*x-b);
    D_con = [-eye(n);D_conl2'];
    
    % r_t
    r_dual = D_obj+D_con'*lambda;       % r_dual
    r_cent = -lambda.*con-onen/t;       % r_cent
    
    if (sqrt(r_dual'*r_dual)<=para.epsilon_feas && eta<=para.epsilon) 
        return;
    end     
    r_t = [r_dual;r_cent];              % r_t
       
    % combaination matrix
    CombMat = [lambda(end)*H_conl2 D_con';-diag(lambda)*D_con -diag(con)];
    
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
        if sum(x_plus<0)==0 && norm(A*x_plus-b,2)^2<=delta
            break;
        else
            s = para.beta*s;
        end
    end     
    
    rt2 = sqrt(r_t'*r_t);
    while 1
        x_plus = x+s*Deltax;
        lambda_plus = lambda+s*Deltalambda;
        
        % Derevative and Hessian of constraint function
        conl2_plus = norm(A*x_plus-b,2)^2-delta;
        
        D_conl2_plus = A'*(A*x_plus-b);
        D_con_plus = [-eye(n);D_conl2_plus'];
        
        % r_t_plus
        r_dual_plus = D_obj+D_con_plus'*lambda_plus;   
        r_cent_plus = -lambda_plus.*[-x_plus;conl2_plus]-onen/t;                   
        r_t_plus = [r_dual_plus;r_cent_plus];     
        
        if sqrt(r_t_plus'*r_t_plus)<=(1-para.alpha*s)*rt2
            break;
        else
            s = para.beta*s;
        end        
        
        if s < 1e-10
            return;
        end
    end
    
    x = x_plus;
    lambda = lambda_plus;     
end







