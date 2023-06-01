close all; clear all;  tic;
% 假如3用户，4导频【1 2 3 4 】
% 用户1 选【1 2】
% 用户2 选【1 3】
% 用户3 选【3 2】
% 这种也算是碰2个
%每个用户导频池中随机选择2个导频，如果有1 个导频被>2用户选择了 ，就存在一次碰撞。
User_M_range=[1:4:18];   %用户数
Pilot_N_range=[20  50 100 150];%导频数
% Pilot_Pool=[1:Pilot_N];%导频池
Pilot_n=2;% 按LTE ,每个用户选择 2列导频

% P_0;%不碰概率
% P_1;%1列任意一列碰概率
% P_2; %2列都碰概率
    K0 = 100;% user number;
    pA =0.1;% activation probability ;
    nbrOfRAblocks =10000;
   newUsers = binornd(K0,pA,[nbrOfRAblocks 1]);    

frame_Num=100;
case_index=0;
for Pilot_N=Pilot_N_range
   % rand('state',12345);  randn('state',98765); % 保证每次SNR循环，初始种子一样
    case_index=case_index+1;
    P_0=zeros(length(User_M_range),1);%不碰概率
    P_1=zeros(length(User_M_range),1);%1列任意一列碰概率
    P_2=zeros(length(User_M_range),1); %2列都碰概率
    for User_index=1:length(User_M_range )
        Pilot_user=[];
        User_M=User_M_range(User_index);
        for frame=1:frame_Num
            for m=1:User_M
                Pilot_user(m,:)=randperm(Pilot_N,Pilot_n);%  两列导频不重复
                %     Pilot_user(m,:)=randi([1,Pilot_N],1,Pilot_n); %两列导频可以重复
            end
            %对每个用户的 两列导频去和其他M-1用户比较，如果完全一样，则2个碰撞。只要存在2
            %% 判断2个导频都碰
            if (length(unique(Pilot_user(:,1)))<User_M)&&(length(unique(Pilot_user(:,2)))<User_M)
                P_2(User_index)=P_2(User_index)+1;
            end

            %%  判断任有1个导频碰
            if (length(unique(Pilot_user(:,1)))<User_M)&&(length(unique(Pilot_user(:,2)))==User_M)||...
                    (length(unique(Pilot_user(:,2)))<User_M)&&(length(unique(Pilot_user(:,1)))==User_M)
                P_1(User_index)=P_1(User_index)+1;
            end     
            %%  判断无导频碰撞
            if (length(unique(Pilot_user(:,1)))==User_M)&&(length(unique(Pilot_user(:,2)))==User_M)
                P_0(User_index)=P_0(User_index)+1;
            end
            
        end
        P_2_frame(:,User_index)=P_2(User_index)./frame_Num ;       %2列都碰概率
        P_1_frame(:,User_index)=P_1(User_index)./frame_Num ;       %其中1列碰概率
        P_0_frame(:,User_index)=P_0(User_index)./frame_Num ;        %不碰概率
    end
    P_2_matrix(case_index,:)=P_2_frame ;       %2列都碰概率
    P_1_matrix(case_index,:)=P_1_frame ;       %其中1列碰概率
    P_0_matrix(case_index,:)=P_0_frame ;        %不碰概率
end
USER_Matrix=repmat(User_M_range,case_index,1);
plot_snr_bler(USER_Matrix,P_1_matrix)
legend('P1-导频N=20','P1-导频N=50','P1-导频N=100','P1-导频N=150','P2-导频N=20','P2-导频N=50','P2-导频N=100','P2-导频N=150','5','6','7','8','9','10')
title('N导频，M用户，每个用户任选两个不同的导频')
xlabel('USER-M'); ylabel('P1-collision');
set(gca,'Fontname','Monospaced'); 

figure
USER_Matrix=repmat(User_M_range,case_index,1);
plot_snr_bler(USER_Matrix,P_2_matrix)
legend('导频N=20','导频N=50','导频N=100','导频N=150')
title('N导频，M用户，每个用户任选两个不同的导频')
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


