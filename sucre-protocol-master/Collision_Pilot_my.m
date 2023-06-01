%%%导频碰撞概率%%%
clc;close all; clear all;
%% set parameters
K0values = [100 200 300 500 1000:5000:21000]; % user number
K0values = [1000:1000:2000]; % user number
pA = 0.0001; % activation probability
nbrOfRAblocks = 10000; % frame
taup_range = [10]; % 导频数目
% taup_range = [10 50 100]; % 导频数目
% two pilot(two slot)
% 1: not repeat  2: repeat
% one pilot(one slot)
% 3: one pilot
method_range = [1 2];
case_index = 0; icount = 0; ucount = 0; mcount = 0;
caseNum_theory = 0;
%% simulation
for taup = taup_range
    rand('state',12345);  randn('state',98765); % 保证每次SNR循环，随机种子相同
    case_index = case_index + 1;
    P_0=zeros(length(K0values),length(method_range)); %不碰概率
    P_1=zeros(length(K0values),length(method_range)); %任意一个碰概率
    P_2=zeros(length(K0values),length(method_range)); %2个碰概率
    nbrOfUsers=zeros(length(K0values),1);
    for indProb = 1:length(K0values)
        disp(['K0 values: ' num2str(indProb) ' out of ' num2str(length(K0values))]);
        K0 = K0values(indProb); % Extract current value of the number of inactive UEs
        actUsers = binornd(K0,pA,[nbrOfRAblocks 1]); % 激活用户数
        for r = 1:nbrOfRAblocks
            for m = 1:length(method_range)
                if method_range(m) == 1 || method_range(m) == 2
                    % Randomize which of the pilots that each of the UEs are using
                    pilotSelections = zeros(actUsers(r),2);
                    if method_range(m) == 1
                        for k = 1:actUsers(r)
                            pilotSelections(k,:)=randperm(taup,2); % 前后导频不可重复
                        end
                    elseif method_range(m) ==2
                        pilotSelections = randi(taup,[actUsers(r) 2]); % 前后导频可重复
                    end
                    
                    % Go through all different UEs
                    if ~isempty(pilotSelections)
                        for k = 1:actUsers(r)
                            pilotSelectmp = pilotSelections;
                            pilotSelectmp(k,:) = [];
                            % 判断当前用户2个都有导频都碰
                            if ismember(pilotSelections(k,1),pilotSelectmp(:,1)) && ismember(pilotSelections(k,2),pilotSelectmp(:,2))
                                P_2(indProb,m)=P_2(indProb,m)+1;
                            end
                            % 判断当前用户任有1个导频碰
                            if (ismember(pilotSelections(k,1),pilotSelectmp(:,1))&&~ismember(pilotSelections(k,2),pilotSelectmp(:,2))) || ...
                                    (~ismember(pilotSelections(k,1),pilotSelectmp(:,1))&&ismember(pilotSelections(k,2),pilotSelectmp(:,2)))
                                P_1(indProb,m)=P_1(indProb,m)+1;
                            end
                            
                            % 判断当前用户无导频碰撞
                            if ~ismember(pilotSelections(k,1),pilotSelectmp(:,1)) && ~ismember(pilotSelections(k,2),pilotSelectmp(:,2))
                                P_0(indProb,m)=P_0(indProb,m)+1;
                            end
                            
                        end
                    end
                elseif method_range(m) == 3
                    pilotSelections = randi(taup,[actUsers(r) 1]);
                    
                    % Go through all different UEs
                    if ~isempty(pilotSelections)
                        for k = 1:actUsers(r)
                            pilotSelectmp = pilotSelections;
                            pilotSelectmp(k,:) = [];
                            % 判断当前用户有无导频碰撞
                            if ismember(pilotSelections(k,1),pilotSelectmp(:,1))
                                P_2(indProb,m)=P_2(indProb,m)+1;
                            else
                                P_0(indProb,m)=P_0(indProb,m)+1;
                            end
                            
                        end
                    end
                end
            end
            
        end
        nbrOfUsers(indProb,1) = sum(actUsers);
    end
    %%
    for method=method_range
        caseNum_theory = caseNum_theory+1;
        for indProb = 1:length(K0values)
            disp(['K0 values: ' num2str(indProb) ' out of ' num2str(length(K0values))]);
            K0 = K0values(indProb); % Extract current value of the number of inactive UEs
            activeUsers_M=0:K0; % 激活用户可取的值
            if  method == 1 % 两个时隙不可重选（文档情况二、第二次选导频不选第一次选过的导频）
                P_activeUsersNum = activeUsersNum_Probabilities (K0,pA,activeUsers_M);% 激活用户服从二项分布的概率
                P0_1 = First_Slot_unCollision_Probabilities(taup,activeUsers_M)';P0_1(find(P0_1>1)) = 1;% 第一个时隙不碰撞的概率
                P0_2 = Second_Slot_unCollision_Probabilities(taup,activeUsers_M)';P0_2(find(P0_2>1)) = 1;% 第二个时隙不碰撞的概率
                P00 = P0_1.*P0_2; % 激活用户数为M条件下，不碰撞概率
                P01 = P0_1.*(1-P0_2)+(1-P0_1).*P0_2; % 激活用户数为M条件下，1个碰撞概率
                P11 = (1-P0_1).*(1-P0_2); % 激活用户数为M条件下，2个碰撞概率
                P_E00 = sum(P00.*P_activeUsersNum); %由联合概率计算边缘概率
                P_E01 = sum(P01.*P_activeUsersNum);
                P_E11 = sum(P11.*P_activeUsersNum);
                probabilities_Matrix(:,indProb) = [P_E00;P_E01;P_E11];
            end
            if method == 2 % 两个时隙可重选（文档情况一、第二次选导频与第一次无关）
                P_activeUsersNum = activeUsersNum_Probabilities (K0,pA,activeUsers_M);% 激活用户二项分布概率
                P0 = First_Slot_unCollision_Probabilities(taup,activeUsers_M)'; P0(find(P0>1)) = 1; %一个时隙不碰撞概率
                P1 = 1-P0;
                P00 = P0.^2; % 激活用户数为M条件下，不碰撞概率  
                P01 = 2*P0.*P1; % 激活用户数为M条件下，1个碰撞概率
                P11 = P1.^2; % 激活用户数为M条件下，2个碰撞概率
                P_E00 = sum(P00.*P_activeUsersNum);
                P_E01 = sum(P01.*P_activeUsersNum);
                P_E11 = sum(P11.*P_activeUsersNum);
                probabilities_Matrix(:,indProb) = [P_E00;P_E01;P_E11];
            end
        end
        if caseNum_theory == 1
            P_theory_Matrix = probabilities_Matrix;
        else
            P_theory_Matrix(size(P_theory_Matrix,1)+1: size(P_theory_Matrix,1)+size(probabilities_Matrix,1),:) = probabilities_Matrix;
        end
    end
    
    P2frame = P_2 ./ nbrOfUsers;
    P1frame = P_1 ./ nbrOfUsers;
    P0frame = P_0 ./ nbrOfUsers;
    
    if case_index == 1
        P2Matrix = P2frame';
        P1Matrix = P1frame';
        P0Matrix = P0frame';
    else
        P2Matrix(size(P2Matrix,1)+1: size(P2Matrix,1)+size(P2frame,2),:) = P2frame';  %2列都碰概率
        P1Matrix(size(P1Matrix,1)+1: size(P1Matrix,1)+size(P1frame,2),:) = P1frame';  %其中1列碰概率
        P0Matrix(size(P0Matrix,1)+1: size(P0Matrix,1)+size(P0frame,2),:) = P0frame';  %不碰概率
    end
end
%% plot
P_simulation_Matrix = [];
for i = 1:size(P0Matrix,1)
    P_simulation_Matrix = [P_simulation_Matrix;P0Matrix(i,:);P1Matrix(i,:);P2Matrix(i,:)];
end
figure
plot_snr_bler(K0values,[P_simulation_Matrix(1:3,:);P_theory_Matrix(1:3,:)])
% set(gca,'Fontname','Monospaced'); set(gca,'FontSize',15);
xlabel('Number of Inactive UEs'); ylabel('Collision-Probability');
title('\tau_p=10，两个导频不可重复')
legend('simulation,E_0_0','simulation,E_0_1','simulation,E_1_1','theory,E_0_0','theory,E_0_1','theory,E_1_1','Location','NorthEastOutside')
figure
plot_snr_bler(K0values,[P_simulation_Matrix(4:6,:);P_theory_Matrix(4:6,:)])
% set(gca,'Fontname','Monospaced'); set(gca,'FontSize',15);
xlabel('Number of Inactive UEs'); ylabel('Collision-Probability');
title('\tau_p=10，两个导频可重复')
legend('simulation,E_0_0','simulation,E_0_1','simulation,E_1_1','theory,E_0_0','theory,E_0_1','theory,E_1_1','Location','NorthEastOutside')
% %% plot
% figure
% plot_snr_bler(K0values,P0Matrix)
% set(gca,'Fontname','Monospaced'); set(gca,'FontSize',15);
% legend('one-pilot, \tau_p=10','two-pilot, \tau_p=10','one-pilot, \tau_p=50','two-pilot, \tau_p=50','one-pilot, \tau_p=100','two-pilot, \tau_p=100','Location','NorthEastOutside')
% title('不碰撞')
% xlabel('Number of Inactive UEs');
% ylabel('Collision-Probability');
% 
% figure
% plot_snr_bler(K0values,P1Matrix)
% set(gca,'Fontname','Monospaced'); set(gca,'FontSize',15);
% legend('one-pilot, \tau_p=10','two-pilot, \tau_p=10','one-pilot, \tau_p=50','two-pilot, \tau_p=50','one-pilot, \tau_p=100','two-pilot, \tau_p=100','Location','NorthEastOutside')
% title('用户任有1个导频碰撞')
% xlabel('Number of Inactive UEs');
% ylabel('Collision-Probability');
% 
% figure
% plot_snr_bler(K0values,P2Matrix)
% set(gca,'Fontname','Monospaced'); set(gca,'FontSize',15);
% legend('one-pilot, \tau_p=10','two-pilot, \tau_p=10','one-pilot, \tau_p=50','two-pilot, \tau_p=50','one-pilot, \tau_p=100','two-pilot, \tau_p=100','Location','NorthEastOutside')
% title('用户发2个导频都碰撞/用户发1个导频碰撞(无法接入)')
% xlabel('Number of Inactive UEs');
% ylabel('Collision-Probability');
%%
function  plot_snr_bler(SNR_Matrix,BLER_Matrix)

LineStyles='-bs -gv -rp --bs --gv --rp --ko -m< -yd --bs -c> --gv --rp --co --m< --kd --y>';
Rows=size(BLER_Matrix,1);
LineStyles=parse(LineStyles);
MarkerSize =6;
LineWidth = 1.5;
for i=1:Rows
    plot(SNR_Matrix,BLER_Matrix(i,:),LineStyles(i,:),'LineWidth',LineWidth,'MarkerSize',MarkerSize);
    hold on;
    grid on;
end
end
function [x] = parse(inStr)
sz=size(inStr);
strLen=sz(2);
x=blanks(strLen);
wordCount=1;
last=0;
for i=1:strLen
    if inStr(i) == ' '
        wordCount = wordCount + 1;
        x(wordCount,:)=blanks(strLen);
        last=i;
    else
        x(wordCount,i-last)=inStr(i);
    end
end

end
%%
function probabilities = activeUsersNum_Probabilities (K0,pA,activeUsers_M)%激活用户数服从二项分布
userValues = activeUsers_M;
probabilities = zeros(length(userValues),1);
for kind = 1:length(userValues)
    probabilities(kind) =  exp(gammaln(K0+1)-gammaln(userValues(kind)+1)-gammaln(K0-userValues(kind)+1)+(userValues(kind)).*log(pA)+(K0-userValues(kind)).*log((1-pA)));
    % probabilities(kind,:) =  exp(  gammaln(K0+1) - gammaln(userValues(kind)+1) - gammaln(K0-userValues(kind)+1) + (userValues(kind)).*log(pA*pP/taup) + (K0-userValues(kind)).*log((1-pA*pP/taup)) );
end
end
%%
function P0_1 = First_Slot_unCollision_Probabilities(taup,activeUsers_M)%
% P0 = (1-1/taup).^activeUsers_M+(activeUsers_M./taup).*((1-1/taup).^(activeUsers_M-1));
P0_1 = (1-1/taup).^(activeUsers_M-1);
end
%%
function P0_2 = Second_Slot_unCollision_Probabilities(taup,activeUsers_M)
% P0 = (1-1/taup).^activeUsers_M+(activeUsers_M./taup).*((1-1/taup).^(activeUsers_M-1));
P0_2 = ((taup-2)/(taup-1)).^(activeUsers_M-1);
end
