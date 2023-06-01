close all; clear all;  tic;
% 假如3用户，4导频【1 2 3 4 】
% 用户1 选【1 2】
% 用户2 选【1 3】
% 用户3 选【3 2】
%用户1两个导频都被碰撞，无法接入；
%而用户2与用户3分别都只有一个导频被碰，故可以接入；另，用户两导频都没被撞亦可接入。
%P00：用户i的两个导频都没有被碰撞;
%P01：用户i的两个导频只有1个被碰撞;
%P2：用户i的两个导频都被碰撞。
K0values = [100 200 300 500 1000:5000:21000]; % 所有用户数
pA = 0.001; % 激活概率
nbrOfRAblocks = 11000; % frame
Pilot_N_range = [10 20]; % 导频数目
for ind=1:length(K0values)
    User_M_range(ind) = binornd(K0values(ind),pA);
end
% User_M_range=[1:2:15];   %用户数
% Pilot_N_range=[20  50 100 150];%导频数
P0=zeros(length(Pilot_N_range),length(User_M_range));%单个时隙内，用户i导频没有被碰撞的概率
P1=zeros(length(Pilot_N_range),length(User_M_range));%单个时隙内，用户i导频被碰撞的概率
C_all=zeros(length(Pilot_N_range),length(User_M_range)); %单个时隙内，用户选导频的总事件数，共N^M种可能
%
for i=1:length(User_M_range)
    if User_M_range(i)==0
        P0(:,i)=1;
        ONE=ones(length(Pilot_N_range),length(User_M_range));
        P1(:,i)=ONE(:,i)-P0(:,i);
    end
    if User_M_range(i)>0
        C_all(:,i)=Pilot_N_range.^(User_M_range(i));
        P0(:,i)=Pilot_N_range'.*(Pilot_N_range'-1).^(User_M_range(i)-1)./C_all(:,i);
        %             P1=ones(length(Pilot_N_range),length(User_M_range))-P0;
        P1(:,i)=ONE(:,i)-P0(:,i);
    end
    
end
P00=P0.^2;
P01=2*P0.*P1;
P2=P1.^2;
P_A=1-P2;%单个用户可以被成功接入的概率
% C_all2=reshape(C_all,length(Pilot_N_range)*length(User_M_range),1);
%Plot the results for DF relaying
figure; grid on; hold on; box on;
%1个时隙，选一个导频，用户的1个导频被碰撞 ，无法接入
plot(User_M_range,P1(1,:),'-bs');
plot(User_M_range,P1(2,:),'-gv');
plot(User_M_range,P1(3,:),'-rp');
plot(User_M_range,P1(4,:),'-ko');
%两个时隙，每个时隙各选一个导频，用户的两个导频都被碰撞 ，无法接入
plot(User_M_range,P2(1,:),'-.bs');
plot(User_M_range,P2(2,:),'-.gv');
plot(User_M_range,P2(3,:),'-.rp');
plot(User_M_range,P2(4,:),'-.ko');
MarkerSize =9;
LineWidth = 1.5;
legend('P1-导频N=20','P1-导频N=50','P1-导频N=100','P1-导频N=150','P2-导频N=20','P2-导频N=50','P2-导频N=100','P2-导频N=150','5','6','7','8','9','10')
title('N导频，M用户，每个用户选1个导频 VS 选两个导频')
xlabel('用户数M'); ylabel('用户无法接入的概率');
set(gca,'Fontname','Monospaced');



