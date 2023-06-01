close all; clear all;  tic;
% ����3�û���4��Ƶ��1 2 3 4 ��
% �û�1 ѡ��1 2��
% �û�2 ѡ��1 3��
% �û�3 ѡ��3 2��
%�û�1������Ƶ������ײ���޷����룻
%���û�2���û�3�ֱ�ֻ��һ����Ƶ�������ʿ��Խ��룻���û�����Ƶ��û��ײ��ɽ��롣
%P00���û�i��������Ƶ��û�б���ײ;
%P01���û�i��������Ƶֻ��1������ײ;
%P2���û�i��������Ƶ������ײ��
K0values = [100 200 300 500 1000:5000:21000]; % �����û���
pA = 0.001; % �������
nbrOfRAblocks = 11000; % frame
Pilot_N_range = [10 20]; % ��Ƶ��Ŀ
for ind=1:length(K0values)
    User_M_range(ind) = binornd(K0values(ind),pA);
end
% User_M_range=[1:2:15];   %�û���
% Pilot_N_range=[20  50 100 150];%��Ƶ��
P0=zeros(length(Pilot_N_range),length(User_M_range));%����ʱ϶�ڣ��û�i��Ƶû�б���ײ�ĸ���
P1=zeros(length(Pilot_N_range),length(User_M_range));%����ʱ϶�ڣ��û�i��Ƶ����ײ�ĸ���
C_all=zeros(length(Pilot_N_range),length(User_M_range)); %����ʱ϶�ڣ��û�ѡ��Ƶ�����¼�������N^M�ֿ���
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
P_A=1-P2;%�����û����Ա��ɹ�����ĸ���
% C_all2=reshape(C_all,length(Pilot_N_range)*length(User_M_range),1);
%Plot the results for DF relaying
figure; grid on; hold on; box on;
%1��ʱ϶��ѡһ����Ƶ���û���1����Ƶ����ײ ���޷�����
plot(User_M_range,P1(1,:),'-bs');
plot(User_M_range,P1(2,:),'-gv');
plot(User_M_range,P1(3,:),'-rp');
plot(User_M_range,P1(4,:),'-ko');
%����ʱ϶��ÿ��ʱ϶��ѡһ����Ƶ���û���������Ƶ������ײ ���޷�����
plot(User_M_range,P2(1,:),'-.bs');
plot(User_M_range,P2(2,:),'-.gv');
plot(User_M_range,P2(3,:),'-.rp');
plot(User_M_range,P2(4,:),'-.ko');
MarkerSize =9;
LineWidth = 1.5;
legend('P1-��ƵN=20','P1-��ƵN=50','P1-��ƵN=100','P1-��ƵN=150','P2-��ƵN=20','P2-��ƵN=50','P2-��ƵN=100','P2-��ƵN=150','5','6','7','8','9','10')
title('N��Ƶ��M�û���ÿ���û�ѡ1����Ƶ VS ѡ������Ƶ')
xlabel('�û���M'); ylabel('�û��޷�����ĸ���');
set(gca,'Fontname','Monospaced');



