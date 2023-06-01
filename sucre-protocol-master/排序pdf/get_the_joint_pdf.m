function [u_value,v_value,joint_pdf] = get_the_joint_pdf(N)


    global mu;
    global sigma2;
    x_range = [-2:0.01:2];              % x的取值范围
    X_pdf = normpdf(x_range,mu,sigma2);
    X_cdf = normcdf(x_range,mu,sigma2);
    u_value = x_range;      % u的取值范围
    v_value = x_range;     % u的取值范围
    numoftype = length(u_value);        % u的个数
    joint_pdf = zeros(numoftype,numoftype);
    for v_index=[1:length(v_value)]
        for u_index=[1:length(u_value)]
            if u_index < v_index
                joint_pdf(v_index,u_index) = N*(N-1)*X_pdf(u_index)*X_pdf(v_index)*((X_cdf(u_index))^(N-2));
                
            else
               % joint_pdf(v_index,u_index) = 0;
               break;
            end
        end
    end




end

