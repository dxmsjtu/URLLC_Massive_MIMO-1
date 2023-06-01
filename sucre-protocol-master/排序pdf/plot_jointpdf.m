close all;
clear all;
clc
% 生成均值为mu,方差为sigma2的正太分布结果。
global mu;global sigma2;global max_value;
mu = 0;
sigma2 = 1;
max_value = 2;
N_range = [10];
joint_pdf = [];

for index = [1:length(N_range)]
   N = N_range(index);
   [u_value,v_value,joint_pdf{index}] = get_the_joint_pdf(N);
end

[u,v]=meshgrid(u_value,v_value);
for plot_index = [1:length(N_range)]
   figure(plot_index)
   mesh(u,v,joint_pdf{plot_index});
   name = ['N=',num2str(N_range(plot_index))];
   legend(name);
   xlabel('u(次大值)');
   ylabel('v(最大值)');
   zlabel('joint_pdf(u,v)');
end
 set(gca,'Fontname','Monospaced'); 
% plot(u_value,joint_pdf{1},'r-',u_value,joint_pdf{2},'b-',u_value,joint_pdf{3},'g-');
% legend('N=10','N=20','N=50');







