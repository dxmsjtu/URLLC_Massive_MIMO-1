%%%导频碰撞概率%%%
close all; clear all; tic; 
% 假如3个用户，4个导频【1 2 3 4】
% 用户1 选【1 2】
% 用户2 选【1 3】
% 用户3 选【3 2】
% 或者
% 假如4个用户，6个导频【1 2 3 4 5 6】
% Pilot_user = [1 2;
%               1 3; 
%               4 5;
%               6 5;];
% 上面这两种都算是碰2个

%% set parameters
K0values = [100 200 300 500 1000:5000:21000]; % user number
pA = 0.001; % activation probability
nbrOfRAblocks = 1000; % frame
taup_range = [10 50 100 150]; % 导频数目
method_range = [1 2 3 ]; % 1: not repeat  2: repeat  3: one pilot
case_index = 0;
%% simulation
for taup = taup_range
    rand('state',12345);  randn('state',98765); % 保证每次SNR循环，随机种子相同
    case_index = case_index + 1;
    P_0=zeros(length(K0values),length(method_range)); %不碰概率
    P_1=zeros(length(K0values),length(method_range)); %1列任意一列碰概率
    P_2=zeros(length(K0values),length(method_range)); %2列都碰概率
    for indProb = 1:length(K0values)
        disp(['K0 values: ' num2str(indProb) ' out of ' num2str(length(K0values))]);
        K0 = K0values(indProb); % Extract current value of the number of inactive UEs
        actUsers = binornd(K0,pA,[nbrOfRAblocks 1]); % 激活用户数
        for r = 1:nbrOfRAblocks
            for method = method_range
                % Randomize which of the pilots that each of the UEs are using
                if method == 1 || method == 2
                    pilotSelections = zeros(actUsers(r),2);
                    if method == 1
                        for m = 1:actUsers(r)
                            pilotSelections(m,:)=randperm(taup,2);
                        end
                    elseif method ==2
                        pilotSelections = randi(taup,[actUsers(r) 2]);
                    end
                    % 判断2列都有导频都碰
                    if (length(unique(pilotSelections(:,1)))<actUsers(r))&&(length(unique(pilotSelections(:,2)))<actUsers(r))
                        P_2(indProb,method)=P_2(indProb,method)+1;
                        P_1(indProb,method)=P_1(indProb,method)+1;
                    end
                    %  判断任有1列有导频碰
                    if (length(unique(pilotSelections(:,1)))<actUsers(r))&&(length(unique(pilotSelections(:,2)))==actUsers(r))||...
                            (length(unique(pilotSelections(:,2)))<actUsers(r))&&(length(unique(pilotSelections(:,1)))==actUsers(r))
                        P_1(indProb,method)=P_1(indProb,method)+1;
                    end
                    % 判断无导频碰撞
                    if (length(unique(pilotSelections(:,1)))==actUsers(r))&&(length(unique(pilotSelections(:,2)))==actUsers(r))
                        P_0(indProb,method)=P_0(indProb,method)+1;
                    end
                    
                elseif method == 3
                    pilotSelections = randi(taup,[actUsers(r) 1]);                    
                    %  判断任有1列有导频碰
                    if (length(unique(pilotSelections(:,1)))<actUsers(r))
                        P_1(indProb,method)=P_1(indProb,method)+1;
                    end
                    % 判断无导频碰撞
                    if (length(unique(pilotSelections(:,1)))==actUsers(r))
                        P_0(indProb,method)=P_0(indProb,method)+1;
                    end
                end
            end
        end
    end    
    P2frame = P_2' ./ nbrOfRAblocks;     P1frame = P_1' ./ nbrOfRAblocks;     P0frame = P_0' ./ nbrOfRAblocks;    
    if case_index == 1
        P2Matrix = P2frame;         P1Matrix = P1frame;        P0Matrix = P0frame;
    else
        P2Matrix(size(P2Matrix,1)+1: size(P2Matrix,1)+size(P2frame,1),:) = P2frame;  %2列都碰概率
        P1Matrix(size(P1Matrix,1)+1: size(P1Matrix,1)+size(P1frame,1),:) = P1frame;  %其中1列碰概率
        P0Matrix(size(P0Matrix,1)+1: size(P0Matrix,1)+size(P0frame,1),:) = P0frame;  %不碰概率
    end
end

%% plot
figure
plot_snr_bler(K0values,P0Matrix)
legend('two-pilot, \tau_p=10','one-pilot, \tau_p=10','two-pilot, \tau_p=50','one-pilot, \tau_p=50','two-pilot, \tau_p=100','one-pilot, \tau_p=100','two-pilot, \tau_p=150','one-pilot, \tau_p=150')
title('不碰撞')
xlabel('Number of Inactive UEs'); 
ylabel('Collision-Probability');

figure
ax1 = gca;
set(ax1,'FontSize',12);
plot_snr_bler(K0values,P1Matrix)
legend('two-pilot, \tau_p=10','one-pilot, \tau_p=10','two-pilot, \tau_p=50','one-pilot, \tau_p=50','two-pilot, \tau_p=100','one-pilot, \tau_p=100','two-pilot, \tau_p=150','one-pilot, \tau_p=150','Location','SouthEast')
title('1列碰撞')
xlabel('Number of Inactive UEs'); 
ylabel('Collision-Probability');

figure
plot_snr_bler(K0values,P2Matrix)
legend('two-pilot, \tau_p=10','one-pilot, \tau_p=10','two-pilot, \tau_p=50','one-pilot, \tau_p=50','two-pilot, \tau_p=100','one-pilot, \tau_p=100','two-pilot, \tau_p=150','one-pilot, \tau_p=150','Location','SouthEast')
title('2列碰撞')
xlabel('Number of Inactive UEs'); 
ylabel('Collision-Probability');

function  plot_snr_bler(SNR_Matrix,BLER_Matrix)

LineStyles='-bs -gv -rp -ko --bs --gv --rp --ko -m< -yd --bs -c> --gv --rp --co --m< --kd --y>';
Rows=size(BLER_Matrix,1);
LineStyles=parse(LineStyles);
MarkerSize =9;
LineWidth = 1.5;
for i=1:Rows
    if i == 2 || i == 5 || i == 8 || i == 11
        continue;
    end
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
