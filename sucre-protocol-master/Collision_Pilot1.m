%%%��Ƶ��ײ����%%%
close all; clear all;
%% set parameters
K0values = [100 200 300 500 1000:5000:21000]; % user number
pA = 0.001; % activation probability
nbrOfRAblocks = 2000; % frame
taup_range = [10 20 ]; % ��Ƶ��Ŀ
% two pilot(two slot)
% 1: not repeat  2: repeat
% one pilot(one slot)
% 3: one pilot
method_range = [ 1 3]; 
case_index = 0; icount = 0; ucount = 0; mcount = 0;
%% simulation
for taup = taup_range
    rand('state',12345);  randn('state',98765); % ��֤ÿ��SNRѭ�������������ͬ
    case_index = case_index + 1;
    P_0=zeros(length(K0values),length(method_range)); %��������
    P_1=zeros(length(K0values),length(method_range)); %����һ��������
    P_2=zeros(length(K0values),length(method_range)); %2��������
    nbrOfUsers=zeros(length(K0values),1);
    for indProb = 1:length(K0values)
        disp(['K0 values: ' num2str(indProb) ' out of ' num2str(length(K0values))]);
        K0 = K0values(indProb); % Extract current value of the number of inactive UEs
        actUsers = binornd(K0,pA,[nbrOfRAblocks 1]); % �����û���
        for r = 1:nbrOfRAblocks
            for m = 1:length(method_range)
                if method_range(m) == 1 || method_range(m) == 2
                    % Randomize which of the pilots that each of the UEs are using
                    pilotSelections = zeros(actUsers(r),2);
                    if method_range(m) == 1
                        for k = 1:actUsers(r)
                            pilotSelections(k,:)=randperm(taup,2); % ǰ��Ƶ�����ظ�
                        end
                    elseif method_range(m) ==2
                        pilotSelections = randi(taup,[actUsers(r) 2]); % ǰ��Ƶ���ظ�
                    end                    
                    % Go through all different UEs
                    if ~isempty(pilotSelections)
                        for k = 1:actUsers(r)
                            pilotSelectmp = pilotSelections;
                            pilotSelectmp(k,:) = []; % ȥ���Լ�
                            % �жϵ�ǰ�û�2�����е�Ƶ����
                            if ismember(pilotSelections(k,1),pilotSelectmp(:,1)) && ismember(pilotSelections(k,2),pilotSelectmp(:,2))
                                P_2(indProb,m)=P_2(indProb,m)+1;
                            end
                            % �жϵ�ǰ�û�����1����Ƶ��
                            if (ismember(pilotSelections(k,1),pilotSelectmp(:,1))&&~ismember(pilotSelections(k,2),pilotSelectmp(:,2))) || ...
                                    (~ismember(pilotSelections(k,1),pilotSelectmp(:,1))&&ismember(pilotSelections(k,2),pilotSelectmp(:,2)))
                                P_1(indProb,m)=P_1(indProb,m)+1;
                            end
                            
                            % �жϵ�ǰ�û��޵�Ƶ��ײ
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
                            % �жϵ�ǰ�û����޵�Ƶ��ײ
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
    
    P2frame = P_2 ./ nbrOfUsers;
    P1frame = P_1 ./ nbrOfUsers;
    P0frame = P_0 ./ nbrOfUsers;
    
    if case_index == 1
        P2Matrix = P2frame';
        P1Matrix = P1frame';
        P0Matrix = P0frame';
    else
        P2Matrix(size(P2Matrix,1)+1: size(P2Matrix,1)+size(P2frame,2),:) = P2frame';  %2�ж�������
        P1Matrix(size(P1Matrix,1)+1: size(P1Matrix,1)+size(P1frame,2),:) = P1frame';  %����1��������
        P0Matrix(size(P0Matrix,1)+1: size(P0Matrix,1)+size(P0frame,2),:) = P0frame';  %��������
    end
end

%% plot
% K0values = [100 200 300 500 1000:5000:21000]; % user number
% pA = 0.001; % activation probability
% K0values =K0values*pA;
figure
plot_snr_bler(K0values,P0Matrix)
set(gca,'Fontname','Monospaced'); set(gca,'FontSize',15);
legend('one-pilot, \tau_p=10','two-pilot, \tau_p=10','one-pilot, \tau_p=50','two-pilot, \tau_p=50','one-pilot, \tau_p=100','two-pilot, \tau_p=100','Location','NorthEastOutside')
title('����ײ')
xlabel('Number of Inactive UEs'); 
ylabel('Collision-Probability');

figure
plot_snr_bler(K0values,P1Matrix)
set(gca,'Fontname','Monospaced'); set(gca,'FontSize',15);
legend('one-pilot, \tau_p=10','two-pilot, \tau_p=10','one-pilot, \tau_p=50','two-pilot, \tau_p=50','one-pilot, \tau_p=100','two-pilot, \tau_p=100','Location','NorthEastOutside')
title('�û�����1����Ƶ��ײ')
xlabel('Number of Inactive UEs'); 
ylabel('Collision-Probability');

figure
plot_snr_bler(K0values,P2Matrix)
set(gca,'Fontname','Monospaced'); set(gca,'FontSize',15);
legend('one-pilot, \tau_p=10','two-pilot, \tau_p=10','one-pilot, \tau_p=50','two-pilot, \tau_p=50','one-pilot, \tau_p=100','two-pilot, \tau_p=100','Location','NorthEastOutside')
title('�û���2����Ƶ����ײ/�û���1����Ƶ��ײ(�޷�����)')
xlabel('Number of Inactive UEs'); 
ylabel('Collision-Probability');

function  plot_snr_bler(SNR_Matrix,BLER_Matrix)

LineStyles='-bs -gv -rp -ko --bs --gv --rp --ko -m< -yd --bs -c> --gv --rp --co --m< --kd --y>';
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
