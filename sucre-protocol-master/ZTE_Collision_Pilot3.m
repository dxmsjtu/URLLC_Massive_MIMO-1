close all; clear all;  tic;
% ����3�û���4��Ƶ��1 2 3 4 ��
% �û�1 ѡ��1 2��
% �û�2 ѡ��1 3��
% �û�3 ѡ��3 2��
% ����Ҳ������2��
%ÿ���û���Ƶ�������ѡ��2����Ƶ�������1 ����Ƶ��>2�û�ѡ���� ���ʹ���һ����ײ��
User_M_range=[1:4:18];   %�û���
Pilot_N_range=[20  50 100 150];%��Ƶ��
% Pilot_Pool=[1:Pilot_N];%��Ƶ��
Pilot_n=2;% ��LTE ,ÿ���û�ѡ�� 2�е�Ƶ

% P_0;%��������
% P_1;%1������һ��������
% P_2; %2�ж�������
    K0 = 100;% user number;
    pA =0.1;% activation probability ;
    nbrOfRAblocks =10000;
   newUsers = binornd(K0,pA,[nbrOfRAblocks 1]);    

frame_Num=100;
case_index=0;
for Pilot_N=Pilot_N_range
   % rand('state',12345);  randn('state',98765); % ��֤ÿ��SNRѭ������ʼ����һ��
    case_index=case_index+1;
    P_0=zeros(length(User_M_range),1);%��������
    P_1=zeros(length(User_M_range),1);%1������һ��������
    P_2=zeros(length(User_M_range),1); %2�ж�������
    for User_index=1:length(User_M_range )
        Pilot_user=[];
        User_M=User_M_range(User_index);
        for frame=1:frame_Num
            for m=1:User_M
                Pilot_user(m,:)=randperm(Pilot_N,Pilot_n);%  ���е�Ƶ���ظ�
                %     Pilot_user(m,:)=randi([1,Pilot_N],1,Pilot_n); %���е�Ƶ�����ظ�
            end
            %��ÿ���û��� ���е�Ƶȥ������M-1�û��Ƚϣ������ȫһ������2����ײ��ֻҪ����2
            %% �ж�2����Ƶ����
            if (length(unique(Pilot_user(:,1)))<User_M)&&(length(unique(Pilot_user(:,2)))<User_M)
                P_2(User_index)=P_2(User_index)+1;
            end

            %%  �ж�����1����Ƶ��
            if (length(unique(Pilot_user(:,1)))<User_M)&&(length(unique(Pilot_user(:,2)))==User_M)||...
                    (length(unique(Pilot_user(:,2)))<User_M)&&(length(unique(Pilot_user(:,1)))==User_M)
                P_1(User_index)=P_1(User_index)+1;
            end     
            %%  �ж��޵�Ƶ��ײ
            if (length(unique(Pilot_user(:,1)))==User_M)&&(length(unique(Pilot_user(:,2)))==User_M)
                P_0(User_index)=P_0(User_index)+1;
            end
            
        end
        P_2_frame(:,User_index)=P_2(User_index)./frame_Num ;       %2�ж�������
        P_1_frame(:,User_index)=P_1(User_index)./frame_Num ;       %����1��������
        P_0_frame(:,User_index)=P_0(User_index)./frame_Num ;        %��������
    end
    P_2_matrix(case_index,:)=P_2_frame ;       %2�ж�������
    P_1_matrix(case_index,:)=P_1_frame ;       %����1��������
    P_0_matrix(case_index,:)=P_0_frame ;        %��������
end
USER_Matrix=repmat(User_M_range,case_index,1);
plot_snr_bler(USER_Matrix,P_1_matrix)
legend('P1-��ƵN=20','P1-��ƵN=50','P1-��ƵN=100','P1-��ƵN=150','P2-��ƵN=20','P2-��ƵN=50','P2-��ƵN=100','P2-��ƵN=150','5','6','7','8','9','10')
title('N��Ƶ��M�û���ÿ���û���ѡ������ͬ�ĵ�Ƶ')
xlabel('USER-M'); ylabel('P1-collision');
set(gca,'Fontname','Monospaced'); 

figure
USER_Matrix=repmat(User_M_range,case_index,1);
plot_snr_bler(USER_Matrix,P_2_matrix)
legend('��ƵN=20','��ƵN=50','��ƵN=100','��ƵN=150')
title('N��Ƶ��M�û���ÿ���û���ѡ������ͬ�ĵ�Ƶ')
xlabel('USER-M'); ylabel('P2-collision');

set(gca,'Fontname','Monospaced'); 
function  plot_snr_bler(SNR_Matrix,BLER_Matrix)

LineStyles='-bs -gv -rp -ko --bs --gv --rp --ko -m< -yd --bs -c> --gv --rp --co --m< --kd --y>';
Rows=size(BLER_Matrix,1);
LineStyles=parse(LineStyles);
MarkerSize =9;
LineWidth = 1.5;
for i=1:Rows
    plot(SNR_Matrix(i,:),BLER_Matrix(i,:),LineStyles(i,:),'LineWidth',LineWidth,'MarkerSize',MarkerSize);
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


