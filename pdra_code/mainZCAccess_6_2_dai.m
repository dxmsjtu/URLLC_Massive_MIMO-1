clc;clear all;close all; 
savefile=1;   
if savefile==1
    filenameTmp ='test_RAcorrChannel_MMSEvsMF_PDRAsim_M256_corr0.6_SINR4_PA0.2_Shift32';%可将仿真关键参数作为文件名
    mkdir_str=strcat('.\Simulation\',filenameTmp);
    mkdir(mkdir_str);%一运行就会在当前文件夹下创建simulation文件夹
    mkdir_str1 =strcat(mkdir_str,'\');
    filenameTmp1= filenameTmp;%配置文件的保存的名字
    mkdir_str =strcat(mkdir_str1,filenameTmp1);
    mkdir_str =strcat(mkdir_str,'.m');
    Save_link1=Save_link_file('mainZCAccess_6_2_dai.m',mkdir_str);%
end
tic 
System_type_range=[1];% 0: 原来每帧用户固定UserNum = K0*pA，  1： 每帧不同用户  伯努利分布   
nbrOfRealizations_range = [1e4]; 
R_range = [1:1:5]+0;%ZC根序列个数 256 512
reMethod_range = [0];comSINR_range = [0:1:0];
pA_range = [20]/10000; M_range = [256]; 
Weight_range = [1]; SNR_range = [1:1:1];
SNR_Two_offset_range = [0];K0values_range = [10000]; 
RepPaper_range = [0]; SequenceLength_range = [839];  %S
Sequence_offset_range = [0];SINR_range = [4:1:4]; %阈值 
PR_ContrFactor_range=[0] ;% 功控因子范围   0db 是 理想功控  ，选2db
L_range = [32];% 9 18 32 64shift个数 
sim = 1;   % 2: 理论数据+仿真数据；1：仿真；0：理论；
ESTIMATOR_range = {'MMSE','LS'}; %what estimator to use [LS, MMSE]
channel_range = [3];% channel = 1,iid ;% channel = 2,corr; % channel = 3,Random Access corr
phaS_range = [pi/9];phaA_upper_range = [pi/3]; % Debug parameter of channel = 2
correlationFactor_range=[ 6]/10; % Debug parameter of channel = 3
method_range = [2]; % method = 1,传一个码 %  method = 2,传两个码
SubcaseNum = 0;  % 方便计算；dxm7311
Default = 0;Dofor = 0; SumIndex =0;RandomSeed =0; cellRadius=250;
tic
for System_type=System_type_range
for nbrOfRealizations = nbrOfRealizations_range for reMethod = reMethod_range for comSINR = comSINR_range
for pA = pA_range  for M = M_range  for Weight = Weight_range for SNR = SNR_range 
for SNR_Two_offset = SNR_Two_offset_range for K0_range = K0values_range
for RepPaper = RepPaper_range for SequenceLength = SequenceLength_range 
for Sequence_offset = Sequence_offset_range for SINR = SINR_range 
for PR_ContrFactor=PR_ContrFactor_range;%  0 (prefect等功率PR=1), 5  10
for L = L_range for ESTIMATOR = ESTIMATOR_range for channel = channel_range 
for phaS = phaS_range for phaA_upper = phaA_upper_range for correlationFactor = correlationFactor_range
for method = method_range
if RandomSeed ==0 rand('state',12345);  randn('state',12345*3);  end% 保证每次SNR循环，初始种子都一样
Para.System_type = System_type;
Para.SubcaseNum = SubcaseNum; Para.Default =Default; Para.Dofor = Dofor;
Para.reMethod=reMethod; Para.comSINR = comSINR;Para.pA = pA ; % 激活概率
Para.Weight = Weight;Para.SNR = SNR;
Para.channel = channel;
Para.phaS = phaS;Para.phaA_upper = phaA_upper;
Para.correlationFactor = correlationFactor;Para.cellRadius = cellRadius;
Para.SINR = SINR ; %阈值
Para.PR_ContrFactor=PR_ContrFactor;
Para.R_range = R_range;%ZC根序列个数
Para.L = L;% shift个数
Para.M = M ;   %Number of BS antennas
Para.ESTIMATOR = ESTIMATOR;
Para.SNR_Two_offset = SNR_Two_offset; 
Para.K0_range = K0_range ; %[100 200 300 500 1000:5000:12000]; % user number
Para.RepPaper = RepPaper;  % 再现文章图5 of [1]
Para.SequenceLength = SequenceLength ;
Para.Sequence_offset = Sequence_offset ; 

if sim == 1 % 全仿真
    DataSim = computeSuccessProbability_sim(method,Para,nbrOfRealizations);% 全仿真
    DataAll = DataSim;
elseif sim == 0 % 全理论
    DataTheory = computeSuccessProbability_theory(method,Para,nbrOfRealizations);
    DataAll = DataTheory;
elseif sim == 2 % 全理论+全仿真
    DataTheory = computeSuccessProbability_theory(method,Para,nbrOfRealizations);
    DataSim = computeSuccessProbability_sim(method,Para,nbrOfRealizations);% 全仿真
    DataAll = [DataTheory;DataSim];
end
SumIndex = SumIndex+1; 
if SumIndex ==1 
    DataMatrix = DataAll; 
else 
    DataMatrix(size(DataMatrix,1)+1: size(DataMatrix,1)+size(DataAll,1),:) = DataAll;
end
% DataMatrix
toc; stop = 1;
if savefile==1
    % simulation文件夹里，保存每次仿真结果  ,成  .mat文件
    strsave= strcat('.\Simulation\',filenameTmp,'\');
    if SumIndex ==1
        filenameTmp1 = strcat(filenameTmp,'.mat');
    end
    strsave= strcat(strsave,filenameTmp1);
    s=['save ' strsave];% 保持.mat 文件，以后仿真结果可以再次确认,以后一定注意可以再次画图。
    eval(s);  
end

end 
end
end
end
end
end 
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
%%
% toc
% DataMatrix
% figure
% plot_snr_bler(R_range,DataMatrix) 
% xlabel('Number Of root sequence ($R$)', 'Interpreter','latex'); ylabel('Success probability $P_{{\rm{MF}}}$', 'Interpreter','latex');
% set(gca,'Fontname','Monospaced'); set(gca,'FontSize',12);
% % set(gca,'YTick',0:0.1:1);
% % title_str = strcat('SINR =',num2str(title_SINR(i)),'dB');
% % title(title_str);set(gca,'Fontname','Monospaced');
% legend('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30');
% h  =gcf; % 获得当前figure 句柄，大家需要用这个模板来画图，仔细调整写出的文章才
% MarkerSize=9; LineWidth =2; LineMethod =1; PlotMethod =1; FontSize=22;
% YLabelFontSize =24;
% FontSize= 24 ; LineWidth = 3;TitleFontSize = 20; LegendFontSize = 24; axis_ratio=1.5; 
% myboldify(h,MarkerSize,YLabelFontSize,FontSize,LineWidth,LegendFontSize,TitleFontSize)
%%
if sim == 2 
if SubcaseNum==1
Index =0;
for i=1:size(DataMatrix,1)/4
    Index= Index+1;
    Ratio(Index,1:size(DataMatrix,2)) = DataMatrix(4*(i-1)+3,:)./DataMatrix(4*(i-1)+1,:);    
    Ratio(Index,size(DataMatrix,2)+1) = mean(Ratio(Index,1:size(DataMatrix,2)));
    Index= Index+1;
    Ratio(Index,1:size(DataMatrix,2)) = DataMatrix(4*(i-1)+4,:)./DataMatrix(4*(i-1)+2,:);    
    Ratio(Index,size(DataMatrix,2)+1) = mean(Ratio(Index,1:size(DataMatrix,2)));
end
  Ratio
end
if SubcaseNum==2
Index =0;
for i=1:size(DataMatrix,1)/6
    Index= Index+1;
    Ratio(Index,1:size(DataMatrix,2)) = DataMatrix(6*(i-1)+4,:)./DataMatrix(6*(i-1)+1,:);    
    Ratio(Index,size(DataMatrix,2)+1) = mean(Ratio(Index,1:size(DataMatrix,2)));
    Index= Index+1;
    Ratio(Index,1:size(DataMatrix,2)) = DataMatrix(6*(i-1)+6,:)./DataMatrix(6*(i-1)+3,:);    
    Ratio(Index,size(DataMatrix,2)+1) = mean(Ratio(Index,1:size(DataMatrix,2)));
end
  Ratio
end
for i=1:size(DataMatrix,1)/2
    Ratio(i,1:size(DataMatrix,2)) = DataMatrix(2*i,:)./DataMatrix(2*(i-1)+1,:);
    Ratio(i,size(DataMatrix,2)+1) = mean(Ratio(i,1:size(DataMatrix,2)));
end
  
  
else
for i=1:size(DataMatrix,1)/6
    Ratio(2*(i-1)+1,1:size(DataMatrix,2)) = DataMatrix(6*(i-1)+4,:)./DataMatrix(6*(i-1)+1,:);
    Ratio(2*(i-1)+1,size(DataMatrix,2)+1) = mean(Ratio(2*(i-1)+1,1:size(DataMatrix,2)));
    Ratio(2*(i),1:size(DataMatrix,2)) = DataMatrix(6*i,:)./DataMatrix(6*i-3,:);
    Ratio(2*(i),size(DataMatrix,2)+1) = mean(Ratio(i,2:size(DataMatrix,2)));
end
end

% Ratio
% [SortedValue, SortedIndex ] =sort(Ratio(:,end).');
% Ratio(SortedIndex(end),:)
% SortedIndex(end)
% %%
% plotFig = 0;
% if plotFig ==1
%      clear all; close all;
%     filenameTmp ='test_cor_channel_151';%test_diff_channel_M test_M test_shift 可将仿真参数作为文件名
%     mkdir_str=strcat('.\Simulation\',filenameTmp);
%     mkdir(mkdir_str);%一运行就会在当前文件夹下创建simulation文件夹
%     mkdir_str1 =strcat(mkdir_str,'\');
%     mkdir_str =strcat(mkdir_str1,filenameTmp);
%     mkdir_str =strcat(mkdir_str,'.m');
%     strsave= strcat('.\Simulation\',filenameTmp,'\');
%     strsave= strcat(strsave,filenameTmp); s=['load ' strsave]; eval(s);% 保持.mat 文件，以后仿真结果可以再次确认,以后一定注意可以再次画图。
%    DataMatrix
%     SNR_Matrix =R_range;
%     CaseNum = length(method_range)*length(L_range)*length(pA_range);
%     for i=1:size(DataMatrix,1)/CaseNum
%         selectedIndex = [(i-1)*CaseNum+1:i*CaseNum];
%         DataMatrix_tmp =DataMatrix(selectedIndex,:);
%         SNR_Matrix_tmp =SNR_Matrix;
% %         title_SINR = repmat(SINR_range,1,size(DataMatrix,1)/CaseNum/length(SINR_range));
%         figure
%         plot_snr_bler(SNR_Matrix_tmp,DataMatrix_tmp);ylim([min(min(DataMatrix_tmp)),max(max(DataMatrix_tmp))])
%         xticks([1:size(DataMatrix_tmp,2)]); xticklabels([1:size(DataMatrix_tmp,2)]);
%  
%         xlabel('Number Of root sequence ($R$)', 'Interpreter','latex'); ylabel('Success probability $P_{{\rm{MF}}}$', 'Interpreter','latex');
%         set(gca,'Fontname','Monospaced'); set(gca,'FontSize',12);
%         %set(gca,'YTick',0:0.1:1);
% %         title_str = strcat('SINR =',num2str(title_SINR(i)),'dB');
% %         title(title_str);set(gca,'Fontname','Monospaced');
%         legend('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30');
%         h  =gcf; % 获得当前figure 句柄，大家需要用这个模板来画图，仔细调整写出的文章才
%         MarkerSize=9; LineWidth =2; LineMethod =1; PlotMethod =1; FontSize=22;
%         YLabelFontSize =24;
%         FontSize= 24 ; LineWidth = 3;TitleFontSize = 20; LegendFontSize = 24; axis_ratio=1.5; %myboldify(h,FontSize,LineWidth,LegendFontSize,TitleFontSize);
%         myboldify(h,MarkerSize,YLabelFontSize,FontSize,LineWidth,LegendFontSize,TitleFontSize)
%     end
%     
%     
% end

%%
function DataMatrix = computeSuccessProbability_theory(method,Para,nbrOfRealizations)
% [1]Jie Ding et al "Analysis of Non-Orthogonal Sequences for Grant-Free RA
% With Massive MIMO"
% clc;clear all;close all;
% method_range = [1 2]; % "1":文章[1];   "2":Proposed
Weight = Para.Weight;SNR = Para.SNR ;
Default = Para.Default;channel = Para.channel;
phaS = Para.phaS ;phaA_upper = Para.phaA_upper;
correlationFactor = Para.correlationFactor ; cellRadius = Para.cellRadius ;
Dofor = Para.Dofor;
comSINR = Para.comSINR;
M = Para.M;   %Number of BS antennas
SNR_Two_offset = Para.SNR_Two_offset; 
K0_range = Para.K0_range; %[100 200 300 500 1000:5000:12000]; % user number
pA = Para.pA; % 激活概率
R_range = Para.R_range;%ZC根序列个数
RepPaper = Para.RepPaper ;  % 再现文章图5 of [1]
L = Para.L;% shift个数
SequenceLength = Para.SequenceLength;
Sequence_offset = Para.Sequence_offset;
SINR = Para.SINR; %阈值
SubcaseNum = Para.SubcaseNum;
reMethod = Para.reMethod ;
if  Default ==1
    M = 128; SNR_Two_offset = 0;
    K0_range = 10000; %[100 200 300 500 1000:5000:12000]; % user number
    pA = 0.003; % 激活概率
    R_range = [1:4]+0;%ZC根序列个数
    RepPaper = 0 ;  % 再现文章图5 of [1]
    L_range = [9 18 32 64];% shift个数
    % L_range = [33 43 53 63 73];% shift个数
    SequenceLength = 139;
    Sequence_offset = 1;
    SINR_range = [3:1:3]; %阈值
    SubcaseNum =1; 
end
sigma2 = 10^(-SNR/10);
rhoR = 10^(SNR/10); PR = rhoR*sigma2;
n = sqrt(sigma2/2)*(randn(M,nbrOfRealizations)+1i*randn(M,nbrOfRealizations));
n_hat = sqrt(1/2)*(randn(M,nbrOfRealizations)+1i*randn(M,nbrOfRealizations));
caseindex = 0;  
for SINR = SINR
for L = L
    P_success = zeros(SubcaseNum,length(R_range));
    caseindex = caseindex +1;
for indProb = 1:length(K0_range)
%     disp(['K0 values: ' num2str(indProb) ' out of ' num2str(length(K0_range))]);
K0 = K0_range(indProb); % Extract current value of the number of inactive UEs
for R = R_range
    if Dofor == 0
        N_range = [0 K0*pA];
    else
        N_range = 0:100;
    end
%     P_N = BernoulliDistribution_Probabilities (K0,pA,N_range);
%     sum(P_N); %激活用户概率
    if method ==1
        Q = R*L; % 导频数
        alpha = 1-L/Q;
    end
    if method ==2
        Q = R*L*(L-1)/2; % 导频数
        alpha = 1-L/Q;
    end
    for N = N_range(2:end)
        %% [1]Jie Ding et al "Analysis of Non-Orthogonal Sequences for Grant-Free RA With Massive MIMO"
        if method  ==1
            if RepPaper==0
                T = min(floor(SequenceLength/10^(SINR/10)),N-1);K_range = 0:T;
            else
                T = min(floor(L/10^(SINR/10)),N-1);K_range = 0:T;
            end
            p(N) = (1-1/Q)^(N-1);
            Pr_one = zeros(length(K_range),1);
            for k = 0:T
                nchoose_k = factorial(N-1)/factorial(k)/factorial(N-1-k);
                v = (1-alpha)^(N-1-k)*alpha^k;
                Pr_one(k+1,1) = nchoose_k*v;
            end
            q(N) = sum(Pr_one);
             %% sim
            if SubcaseNum == 2
                if channel == 1%uncorrelated Rayleigh fading channel
                    h_iid = sqrt(1/2)*(randn(M,nbrOfRealizations,N)+1i*randn(M,nbrOfRealizations,N));
                    h = h_iid;
                end
                if channel == 2
                    multipathNum = M/2; d = 1/2;
                    h_corr = corr_channel(M,nbrOfRealizations,N,multipathNum,d,phaS,phaA_upper);%correlated Rayleigh fading channel
                    h = h_corr;
                end
                if channel == 3 %Random Access correlated Rayleigh fading channel
                    h_corrRA = RAcorr_channel(M,nbrOfRealizations,N,correlationFactor,cellRadius);
                    h = h_corrRA;
                end 
                K_range_sim = 0:N-1;h1 = h(:,:,1);
                for K = K_range_sim
                    if K>=1
                        if RepPaper == 1
                            e1 = sum(h(:,:,2:K+1),3)/sqrt(L)+n_hat/sqrt(L*rhoR);
                        else
                            e1 = sum(h(:,:,2:K+1),3)/sqrt(SequenceLength)+n_hat/sqrt(SequenceLength*rhoR);
                        end
                    else
                        if RepPaper == 1
                            e1 = n_hat/sqrt(L*rhoR);
                        else
                            e1 = n_hat/sqrt(SequenceLength*rhoR);
                        end
                    end
                    if comSINR == 0% 公式（1）
                        h_hat = h1+e1;
                        Numerator = PR*abs(sum(conj(h_hat).*h1,1)).^2;
                        a0 = repmat(conj(h_hat),[1 1 N-1]).*h(:,:,2:N);b0 = sum(a0,1); c0 = sum(abs(b0).^2,3);
                        Denominator1 = PR*c0;Denominator2 = sigma2*abs(sum(conj(h_hat),1)).^2;SINR_sim = Numerator./(Denominator1+Denominator2);
                    end  
                    if comSINR == 1% 公式（1）
                        h_hat = h1+e1;
                        Numerator = PR*abs(sum(conj(h_hat).*h1,1)).^2;
                        a0 = repmat(conj(h_hat),[1 1 N-1]).*h(:,:,2:N);b0 = sum(a0,1); c0 = sum(abs(b0).^2,3);
                        Denominator1 = PR*c0;Denominator2 = abs(sum(conj(h_hat).*n,1)).^2;SINR_sim = Numerator./(Denominator1+Denominator2);
                    end
                    if comSINR ==2% 公式（2）
                        Denominator1 = PR*abs(sum(conj(e1).*h1,1)).^2;
                        a1 = repmat(conj(h1),[1 1 N-1]).*h(:,:,2:N);b1 = sum(a1,1); c1 = sum(abs(b1).^2,3);
                        Denominator2 = PR*c1;
                        a2 = repmat(conj(e1),[1 1 N-1]).*h(:,:,2:N); b2 = sum(a2,1); c2 = sum(abs(b2).^2,3);
                        Denominator3 = PR*c2;
                        Denominator4 = abs(sum(conj(h1).*n,1)).^2;
                        Denominator5 = abs(sum(conj(e1).*n,1)).^2;
                        Numerator = PR*abs(sum(conj(h1).*h1,1)).^2;
                        Denominator = Denominator1+Denominator2+Denominator3+Denominator4+Denominator5;
                        SINR_sim = Numerator./(Denominator);
                    end
                    threshold = 10^(SINR/10);
                    P_SINR(K+1,1) = length(find(  SINR_sim >=threshold ))/nbrOfRealizations;
                end
                Pr_one_sim = zeros(length(K_range),1);
                for k = K_range_sim
                    nchoose_k = factorial(N-1)/factorial(k)/factorial(N-1-k);
                    Pr_one_sim(k+1,1) = nchoose_k*(1-alpha)^(N-1-k)*alpha^k;
                end
                q_sim(N) = sum(Pr_one_sim.*P_SINR); % 原始
            end
        end
        %% 每个用户选择两个正交序列
        if method == 2
            T = min(floor(1/4*SequenceLength/10^(SINR/10)),N-1);K_range = 0:T;
            p(N) = (1-1/Q)^(N-1);
            Pr_two = zeros(length(K_range),1);
            for k = K_range
                %nchoosek(N-1,k)
                if k > 150
                    Pr_two(k+1,1) = factorial(N-1)/factorial(k)/factorial(N-1-k)*((L*(L-1)/2-1))^(N-1-k)*((Q-L*(L-1)/2))^k/((Q-1))^(N-1);
                else
                    %Pr_two(k+1,1) = nchoosek(N-1,k)*((L*(L-1)/2-1))^(N-1-k)*((Q-L*(L-1)/2))^k/((Q-1))^(N-1);
                    Pr_two(k+1,1) = factorial(N-1)/factorial(k)/factorial(N-1-k)*((L*(L-1)/2-1))^(N-1-k)*((Q-L*(L-1)/2))^k/((Q-1))^(N-1);
                end
            end
            if Weight==0
                q(N) = sum(Pr_two);
            else
                M_range = N - K_range;
                [P0,P1] = abc(M_range,L);
                Pr_0and1 =  P0+P1;
                q(N) = sum(Pr_two.*Pr_0and1);
            end
             %% sim
            if SubcaseNum == 2
                if channel == 1%uncorrelated Rayleigh fading channel
                    h_iid = sqrt(1/2)*(randn(M,nbrOfRealizations,N)+1i*randn(M,nbrOfRealizations,N));
                    h = h_iid;
                end
                if channel == 2 %correlated Rayleigh fading channel
                    multipathNum = M/2; d = 1/2;
                    h_corr = corr_channel(M,nbrOfRealizations,N,multipathNum,d,phaS,phaA_upper);
                    h = h_corr;
                end
                if channel == 3 %Random Access correlated Rayleigh fading channel
                    h_corrRA = RAcorr_channel(M,nbrOfRealizations,N,correlationFactor,cellRadius);
                    h = h_corrRA;
                end
%                 h = sqrt(1/2)*(randn(M,nbrOfRealizations,N)+1i*randn(M,nbrOfRealizations,N));%uncorrelated Rayleigh fading channel
                K_range_sim = 0:N-1;h1 = h(:,:,1);
                for K = K_range_sim
                    if reMethod == 1
                    if K>=1
                        if RepPaper == 1
                            e1 = sum(h(:,:,2:K+1),3)/sqrt(L)+n_hat/sqrt(L*rhoR);
                        else
                            e1 = sum(h(:,:,2:K+1),3)/sqrt(SequenceLength)+n_hat/sqrt(SequenceLength*rhoR);
                        end
                    else
                        if RepPaper == 1
                            e1 = n_hat/sqrt(L*rhoR);
                        else
                            e1 = n_hat/sqrt(SequenceLength*rhoR);
                        end
                    end
                    if comSINR == 0% 公式（1）
                        h_hat = h1+e1;
                        Numerator = PR*abs(sum(conj(h_hat).*h1,1)).^2;
                        a0 = repmat(conj(h_hat),[1 1 N-1]).*h(:,:,2:N);b0 = sum(a0,1); c0 = sum(abs(b0).^2,3);
                        Denominator1 = PR*c0;Denominator2 = sigma2*abs(sum(conj(h_hat),1)).^2;SINR_sim_0 = Numerator./(Denominator1+Denominator2);
                    end
                    threshold = 10^(SINR/10);
                    P_SINR_0(K+1,1) = length(find(  SINR_sim_0 >= threshold ))/nbrOfRealizations;
                    end
                    if reMethod ==0
                        if K>=1
                            if RepPaper == 1
                                g1_0 = sqrt(PR*L)*h1+2*sqrt(PR)*sum(h(:,:,2:K+1),3)+n;
                                g1_1 = sqrt(PR*L/2)*h1+sqrt(2*PR)*sum(h(:,:,2:K+1),3)+n;
                            else
                                g1_0 = sqrt(PR*SequenceLength)*h1+2*sqrt(PR)*sum(h(:,:,2:K+1),3)+n;
                                g1_1 = sqrt(PR*SequenceLength/2)*h1+sqrt(2*PR)*sum(h(:,:,2:K+1),3)+n;
                            end
                        else
                            if RepPaper == 1
                                g1_0 = sqrt(PR*L)*h1+n;
                                g1_1 = sqrt(PR*L/2)*h1+n;
                            else
                                g1_0 = sqrt(PR*SequenceLength)*h1+n;
                                g1_1 = sqrt(PR*SequenceLength/2)*h1+n;
                            end
                        end
                        if comSINR == 0% 公式（1）
                            Numerator_0 = PR*abs(sum(conj(g1_0).*h1,1)).^2;
                            a0 = repmat(conj(g1_0),[1 1 N-1]).*h(:,:,2:N);b0 = sum(a0,1); c0 = sum(abs(b0).^2,3);
                            Denominator1_0 = PR*c0;Denominator2_0 = sigma2*abs(sum(conj(g1_0),1)).^2;
                            SINR_sim_0 = Numerator_0./(Denominator1_0+Denominator2_0);
                            Numerator_1 = PR*abs(sum(conj(g1_1).*h1,1)).^2;
                            a0 = repmat(conj(g1_1),[1 1 N-1]).*h(:,:,2:N);b0 = sum(a0,1); c0 = sum(abs(b0).^2,3);
                            Denominator1_1 = PR*c0;Denominator2_1 = sigma2*abs(sum(conj(g1_1),1)).^2;
                            SINR_sim_1 = Numerator_1./(Denominator1_1+Denominator2_1);
                        end
                        threshold = 10^(SINR/10);
                        P_SINR_0(K+1,1) = length(find(  SINR_sim_0 >= threshold ))/nbrOfRealizations;
                        P_SINR_1(K+1,1) = length(find(  SINR_sim_1 >= threshold ))/nbrOfRealizations;
                    end
                end
                Pr_two_sim = zeros(length(K_range),1);
                for k = K_range_sim
                    nchoose_k = factorial(N-1)/factorial(k)/factorial(N-1-k);
                    Pr_two_sim(k+1,1) = nchoose_k*((L*(L-1)/2-1))^(N-1-k)*((Q-L*(L-1)/2))^k/((Q-1))^(N-1);
                end
                if Weight==0
                    q_sim(N) = sum(Pr_two_sim.*P_SINR_0);
                else
                    M_range_sim = N - K_range_sim;
                    [P0_sim,P1_sim] = abc(M_range_sim,L);
                    Pr_0and1_sim =  P0_sim+P1_sim;
                    if reMethod == 1
                        q_sim(N) = sum(Pr_two_sim.*(P_SINR_0.*Pr_0and1_sim));
                    end
                    if reMethod == 0
                        q_sim(N) = sum(Pr_two_sim.*(P_SINR_0.*P0_sim + P_SINR_1.*P1_sim));
                    end
                end
                
            end
        end
    end
    f = p.*q;
    N_average = round(K0*pA);% 平均接入用户数
    P_success(1,R) = f(N_average);% 先对N取平均，再计算概率
    if SubcaseNum == 2
        g = p.*q_sim; 
        P_success(2,R) = g(N_average);
    end
end
end
if caseindex == 1
    DataMatrix = P_success;
else
    DataMatrix(size(DataMatrix,1)+1:size(DataMatrix,1)+size(P_success,1),:) = P_success;
end
end
end
end
function y=factorial(x)
y=1;
while(x~=0)
    y=y*x;
    x=x-1;
end
end
function y=permutation(x,m)
y=1;
for i=0:m-1
    y=y*(x-i);
end
end

function [P0,P1] = abc(M_range,L_range)
% M_range 用户数
% L_range 导频数
caseindex =0;
for L = L_range
for M = M_range
caseindex =caseindex + 1;
A = L-2; B = A; D = (L-2)*(L-3)/2; 
E = 1; G = L*(L-1)/2;
P1 = 0;P2 = 0;
for T = 1:M-1
    nchoose_T = factorial(M-1)/factorial(T)/factorial(M-1-T);
    P1(T) = 2*nchoose_T*D^(M-1-T)*A^T/(G-E)^(M-1);
    P2(T) = nchoose_T*B^(M-1-T)*A^T/(G-E)^(M-1);
end
P3 = 0;
for z = 1:M-3
    for x = 1:M-2-z
        y = M-1-x-z;
        nchoose_x = factorial(M-1)/factorial(x)/factorial(M-1-x);
        nchoose_y = factorial(M-1-x)/factorial(y)/factorial(M-1-x-y);
        a = nchoose_x*A^x;
        b = nchoose_y*B^y; 
%         a = nchoosek(M-1,x)*A^x;
%         b = nchoosek(M-1-x,y)*B^y;
        c = D^z;
        P3 = P3+a*b*c/(G-E)^(M-1);
    end
end
P00 = (D/(G-E))^(M-1);
P01 = sum(P1);
P11 = sum(P2(1:end-1))+P3;
P = [P00 P01 P11 P00+P01 P00+P01+P11];
if caseindex == 1
    Data = P;
else
    Data(size(Data,1)+1:size(Data,1)+size(P,1),:) = P;
end
end
end
P0 = Data(:,1);
P1 = Data(:,2);
end
%%
function P_success = computeSuccessProbability_sim(method,Para,nbrOfRealizations)
Default = Para.Default; M = Para.M;   %Number of BS antennas
K0_range = Para.K0_range; %[100 200 300 500 1000:5000:12000]; % user number
pA = Para.pA; % 激活概率
R_range = Para.R_range;%ZC根序列个数
L = Para.L;% shift个数
SequenceLength = Para.SequenceLength;
ESTIMATOR = Para.ESTIMATOR;
SINR = Para.SINR; %阈值
PR_Control_index=Para.PR_ContrFactor; % 功控因子
SNR = Para.SNR ;channel = Para.channel;
phaS = Para.phaS ;phaA_upper = Para.phaA_upper;
correlationFactor = Para.correlationFactor ; cellRadius = Para.cellRadius ;
System_type=Para.System_type ;
if  Default ==1
    
end
sigma2 = 10^(-SNR/10); % 噪声功率
rhoR = 10^(SNR/10);% 发射功率
PR = rhoR*sigma2;  
multipathNum = M/2;%path number 
d=1/2;% antenna distance
P_success = zeros(1,length(R_range)); % 初始化接入成功概率
PR_ContrFactor =10.^(-[-PR_Control_index:PR_Control_index]/10);%  
for indProb = 1:length(K0_range)
%     disp(['K0 values: ' num2str(indProb) ' out of ' num2str(length(K0_range))]);
    K0 = K0_range(indProb); % Extract current value of the number of inactive UEs
%     UserNum = K0*pA; % 激活用户数
if System_type==0;newUsersNum = repmat(K0*pA,nbrOfRealizations,1); 
elseif System_type==1;newUsersNum = binornd(K0,pA,[nbrOfRealizations 1]);end % 激活用户数

% ZC导频构造
for R = R_range
pilot_pool = zeros(SequenceLength,L,R);
for RootIndex = 1:R
    zcRootSequence = CreatZC(SequenceLength, RootIndex);
    zcShiftSequence = CreatZCcirshift(zcRootSequence,SequenceLength,L);
    pilot_pool(:,:,RootIndex) = zcShiftSequence;
end

if method ==1;    Pilot_combination = (1:L)';           end
if method ==2;    Pilot_combination = nchoosek(1:L,2);  end % 从集合 [1:L] 中选取 2 个元素的所有组合
Number = 0;warning('off');
for r = 1:nbrOfRealizations 
    UserNum=newUsersNum(r); % 每一帧激活用户数 
%     if PR_Control_index==0
%         PR_ContrFactor_user=ones(1,UserNum);
%     else
%     PR_ContrFactor =10.^(-[-PR_Control_index:(PR_Control_index*2/UserNum):PR_Control_index]/10);% 
%     PR_ContrFactor_user=PR_ContrFactor(1:UserNum);% 每一帧用户的功控因子
%     end
    PR_ContrFactor_user=PR_ContrFactor(randi([1,length(PR_ContrFactor)],1,UserNum));% 每一帧用户的功控因子
    if method == 1  
        rootSelections = zeros(UserNum,1);pilotSelection = zeros(UserNum,1);
        rootSelections(1) = randperm(R,1);
        index_SameRoot = randperm(size(Pilot_combination,1),1); pilotSelection(1) = Pilot_combination(index_SameRoot,:);
        Pilot_combination2 = Pilot_combination;
%         Pilot_combination2(index) = [];% 删除与用户1相同的导频
        for user_n = 2:UserNum
            rootSelections(user_n,1) = randperm(R,1);
            if rootSelections(user_n,1) == rootSelections(1)
                % 同根用户与用户1选择导频不同                
                pilotSelection(user_n,:) = Pilot_combination2(randperm(size(Pilot_combination2,1),1),:);
            else
                pilotSelection(user_n,:) = Pilot_combination(randperm(size(Pilot_combination,1),1),:);
            end
        end
    end
    if method == 2  
        rootSelections = zeros(UserNum,1);pilotSelection = zeros(UserNum,2);
        % 用户1 选导频
        rootSelections(1) = randperm(R,1); % 根序列选择
        index_SameRoot = randperm(size(Pilot_combination,1),1); pilotSelection(1,:) = Pilot_combination(index_SameRoot,:); % 两个子导频选择
        Pilot_combination2 = Pilot_combination;
%         Pilot_combination2(index,:) = [];% 删除与用户1相同的导频

        for user_n = 2:UserNum
            rootSelections(user_n,1) = randperm(R,1);
            if rootSelections(user_n,1) == rootSelections(1) 
                pilotSelection(user_n,:) = Pilot_combination2(randperm(size(Pilot_combination2,1),1),:);
            else
                pilotSelection(user_n,:) = Pilot_combination(randperm(size(Pilot_combination,1),1),:);
            end
        end
    end
    PilotIndex = [pilotSelection rootSelections]; 
    if channel == 1 %uncorrelated Rayleigh fading channel
        h_iid = (randn(M,UserNum)+1j*randn(M,UserNum)).*(sqrt(1/2));
        h = h_iid;
    end
    if channel == 2 %correlated Rayleigh fading channel
        h_corr = corr_channel(M,1,UserNum,multipathNum,d,phaS,phaA_upper);
        h = h_corr;
    end
    if channel == 3 %Random Access correlated Rayleigh fading channel，这里随机生成用户位置
        [h_corrRA, Rmatrix] = RAcorr_channel(M,1,UserNum,correlationFactor,cellRadius);
        Rmatrix = squeeze(Rmatrix(:,:,1,:));
        h = h_corrRA;
    end
    Y=zeros(M,SequenceLength); Noise = sqrt(sigma2/2)*(randn(M,SequenceLength)+1i*randn(M,SequenceLength));
for user_n=1:UserNum
%     PilotIndex(user_n,:);
    c = pilot_pool(:,pilotSelection(user_n,:),rootSelections(user_n));
    if method == 1
        s = c(:,1);
        Y=Y+sqrt(PR_ContrFactor_user(user_n))*sqrt(PR)*h(:,user_n)*(s).';
    end
    if method == 2    
        s = (c(:,1)+c(:,2))/sqrt(2);
        Y=Y+sqrt(PR_ContrFactor_user(user_n))*sqrt(PR)*h(:,user_n)*(s).'; 
    end
end
    Y=Y+Noise;
    firstUser_Pilot = PilotIndex(1,:);
    if method == 1
        first = firstUser_Pilot(1);root = firstUser_Pilot(2); 
        c1 = pilot_pool(:,first,root);
        g1 = Y*conj(c1)/norm(c1)^2;
    end
    if method ==2 
        first = firstUser_Pilot(1);second = firstUser_Pilot(2);root = firstUser_Pilot(3);
        c1 = pilot_pool(:,first,root);c2 = pilot_pool(:,second,root);
        c1 = c1/sqrt(2);c2 = c2/sqrt(2);
        index_SameRoot = find(PilotIndex(:,3)== root);% 同根用户索引
        index_diffRoot = find(PilotIndex(:,3)~= root);% 不同根用户索引
        pilotSelectSameRoot = pilotSelection(index_SameRoot(2:end),:); % 同根用户所选导频
        %判断pilot中有1且没2
        component1 = (pilotSelectSameRoot(:, 1) == first & pilotSelectSameRoot(:, 2) ~= second)|(pilotSelectSameRoot(:, 2) == first & pilotSelectSameRoot(:, 1) ~= second);
        Li1 = any(component1);   % 是否存在这种用户，返回逻辑值，存在为1，不存在为0
        index_component1 = find(component1 == 1);   % 对应用户索引
        %判断pilot中有2且没1
        component2 = (pilotSelectSameRoot(:, 1) == second & pilotSelectSameRoot(:, 2) ~= first)|(pilotSelectSameRoot(:, 2) == second & pilotSelectSameRoot(:, 1) ~= first);
        Li2 = any(component2);  % 是否存在这种用户，返回逻辑值
        index_component2 = find(component2 == 1);  % 对应用户索引
        index_component = vertcat(index_component1, index_component2);
                %Li1 = ismember(first,pilotSelectSameRoot);
                %Li2 = ismember(second,pilotSelectSameRoot);%判断pilot中有没有2，有Lib=1,没有Lib=0
        %判断（1,2）有没有在pilot的同一行中出现
        [Li_complete,index_complete] = ismember([first second],pilotSelectSameRoot,'rows');
        if Li1==0 && Li2==0 && Li_complete==0% 同根用户与第一个用户不碰撞
            y1 = Y*conj(c1+c2)/norm(c1+c2)^2;
            if strcmp(ESTIMATOR, 'MMSE')
                P = PR_ContrFactor_user(1)*PR; % 不同用户功率不一样的话要改一下
                Q = P*SequenceLength*Rmatrix(:,:,1)+sum(4*P*Rmatrix(:,:,index_diffRoot),3)+sigma2*eye(M);
                R1Qinv = Rmatrix(:,:,1)/ Q;
                g1 = sqrt(P)*sqrt(SequenceLength)*R1Qinv*y1;
            else
                g1 = y1;
            end
            
        end
        if Li1==1 && Li2==0 && Li_complete==0%同根用户与第一个用户碰撞c1
            y1 = Y*conj(c2)/norm(c2)^2;
            if strcmp(ESTIMATOR, 'MMSE')
                P = PR_ContrFactor_user(1)*PR;
                Q = P*(SequenceLength/2)*Rmatrix(:,:,1)+sum(2*P*Rmatrix(:,:,index_diffRoot),3)+sigma2*eye(M);
                R1Qinv = Rmatrix(:,:,1)/ Q;
                g1 = sqrt(P)*sqrt(SequenceLength/2)*R1Qinv*y1;
            else
                g1 = y1;
            end
        end
        if Li1==0 && Li2==1 && Li_complete==0%同根用户与第一个用户碰撞c2
            y1 = Y*conj(c1)/norm(c1)^2;
            if strcmp(ESTIMATOR, 'MMSE')
                P = PR_ContrFactor_user(1)*PR;
                Q = P*(SequenceLength/2)*Rmatrix(:,:,1)+sum(2*P*Rmatrix(:,:,index_diffRoot),3)+sigma2*eye(M);
                R1Qinv = Rmatrix(:,:,1)/ Q;
                g1 = sqrt(P)*sqrt(SequenceLength/2)*R1Qinv*y1;
            else
                g1 = y1;
            end
        end
        if Li_complete==1 && Li1==0 && Li2==0 %同根用户与第一个用户全碰撞
            y1 = Y*conj(c1+c2)/norm(c1+c2)^2;
            if strcmp(ESTIMATOR, 'MMSE')
                P = PR_ContrFactor_user(1)*PR;
                Q = P*SequenceLength*Rmatrix(:,:,1)+P*SequenceLength*sum(Rmatrix(:,:,index_complete),3)+4*P*sum(Rmatrix(:,:,index_SameRoot),3)+sigma2*eye(M);
                R1Qinv = Rmatrix(:,:,1)/ Q;
                g1 = sqrt(P)*sqrt(SequenceLength)*R1Qinv*y1;
            else
                g1 = y1;
            end
        end
        if (Li1==1 || Li2==1) && Li_complete==1 %同根用户与第一个用户全碰撞,且还存在部分碰撞
            y1 = Y*conj(c1+c2)/norm(c1+c2)^2;
            if strcmp(ESTIMATOR, 'MMSE')
                P = PR_ContrFactor_user(1)*PR;
                Q = P*SequenceLength*Rmatrix(:,:,1)+P*SequenceLength*sum(Rmatrix(:,:,index_complete),3)+...
                    P*(SequenceLength/4)*sum(Rmatrix(:,:,index_component),3)+sum(4*P*Rmatrix(:,:,index_SameRoot),3)+sigma2*eye(M);
                R1Qinv = Rmatrix(:,:,1)/ Q;
                g1 = sqrt(P)*sqrt(SequenceLength)*R1Qinv*y1;
            else
                g1 = y1;
            end
        end  
        if Li1==1 && Li2==1 && Li_complete==0 %同根用户存在两部分碰撞
            y1 = Y*conj(c1+c2)/norm(c1+c2)^2;
            if strcmp(ESTIMATOR, 'MMSE') 
                P = PR_ContrFactor_user(1)*PR;
                Q = P*SequenceLength*Rmatrix(:,:,1)+sum(4*P*Rmatrix(:,:,index_SameRoot),3)+...
                    P*(SequenceLength/4)*sum(Rmatrix(:,:,index_component),3)+sigma2*eye(M);
                R1Qinv = Rmatrix(:,:,1)/ Q;
                g1 = sqrt(P)*sqrt(SequenceLength)*R1Qinv*y1;
            else
                g1 = y1;
            end
        end  
    end
    h1 = h(:,1);
    Numerator = PR_ContrFactor_user(1)*PR*abs(sum(conj(g1).*h1))^2;% PR*abs(sum(conj(h_hat).*h1,1)).^2;
    a0 = repmat(conj(g1),[1 UserNum-1]).*h(:,2:UserNum); b0 = sum(a0,1); c0 = sum(PR_ContrFactor_user(2:end).*abs(b0).^2);
    Denominator1 = PR*c0;
    Denominator2 = sum((abs(conj(g1)).^2))*sigma2;
%     Denominator2 = abs(sum(conj(g1),1)).^2*sigma2;
    SINR_sim = Numerator/(Denominator1+Denominator2);
    threshold = 10^(SINR/10);
    if SINR_sim > threshold
        Number = Number+1;
    end
    if mod(r,1e3) == 0
        disp([num2str(r) 'of' num2str(nbrOfRealizations)]);
    end
end
P_success(R) = Number/nbrOfRealizations;

disp(['P_success of R = ' num2str(R) ' with ' char(ESTIMATOR) '=' num2str(P_success(R))]);
end
end
end

 
function [zcRootSequence] = CreatZC(SequenceLength,RootIndex)%生成ZC序列根序列
%生成ZC序列根序列
if nargin == 1
    RootIndex = SequenceLength-1;
end
n = (0:SequenceLength-1)';
if mod(SequenceLength, 2)==0
    zcRootSequence=exp(-1j*pi*RootIndex/SequenceLength*(n.*n));
else
    zcRootSequence=exp(-1j*pi*RootIndex/SequenceLength*(n.*(n+1)));
end
end
function [zcShiftSequence] = CreatZCcirshift(zcRootSequence,SequenceLength,shiftNum)%生成ZC序列循环移位序列
%生成ZC序列循环移位序列
Ncs = floor(SequenceLength/shiftNum);%循环移位间隔
zcShiftSequence = zeros(SequenceLength,shiftNum);
for k = 0:shiftNum-1
v =k*Ncs;
zcShiftSequence(:,k+1) = circshift(zcRootSequence,v,1);
end
end
%% [2]Success Probability of Grant-Free Random Access With Massive MIMO
function h_corr = corr_channel(M,nbrOfRealizations,UserNum,Q,d,phaS,phaA_upper)%correlated Rayleigh fading channel
% M BS天线数
% UserNum 用户数
% d 天线距离 
% Q 多径数
% phaA_upper  the azimuth angle of the UE location 上界
% phaS  the angle spread
h = zeros(M,nbrOfRealizations,UserNum);
for user_n=1:UserNum
    for r = 1:nbrOfRealizations
        v=(randn(Q,1)+1i*randn(Q,1))/sqrt(2);
        % And φA and φS are defined as the azimuth angle of the UE location and the angle spread
        phaA=2*phaA_upper*rand(1)-phaA_upper;
        AoA = phaS*rand(Q,1)+(phaA-phaS/2);
        a = zeros(M,Q);
        for q = 1:length(AoA)
            a(:,q) = (sqrt(1/Q)*exp(-1i*2*pi*d*(0:M-1)*cos(AoA(q)))).';
        end
        hn = a*v;
        h(:,r,user_n) = hn;
    end
end
if nbrOfRealizations == 1
    h_corr = zeros(M,UserNum);
    for user_n=1:UserNum
        h_corr(:,user_n) = h(:,:,user_n);
    end
else
    h_corr = h;
end
end
%% [3]A Random Access Protocol for Pilot Allocation in Crowded Massive MIMO Systems  随机接入相关信道 
function [h_corr,R] = RAcorr_channel(M,nbrOfRealizations,UserNum,correlationFactor,cellRadius)%correlated Rayleigh fading channel
% M BS天线数
% UserNum 用户数
% correlationFactor 相关因子
% cellRadius 小区半径
% userLocations = generatePointsHexagon([1,nbrOfRealizations,UserNum],cellRadius,0.1*cellRadius);
% userAngles = angle(userLocations);
userAngles = 2*pi*rand([1,nbrOfRealizations,UserNum])-pi;
v = (randn(M,nbrOfRealizations,UserNum)+1i*randn(M,nbrOfRealizations,UserNum))/sqrt(2);
h = zeros(M,nbrOfRealizations,UserNum);
R = zeros(M,M,nbrOfRealizations,UserNum);
for j = 1:nbrOfRealizations
    for userInd = 1:UserNum         
        R(:,:,j,userInd) = toeplitz((correlationFactor*exp(1i*sin(userAngles(1,j,userInd)))).^(0:M-1));% [1]. Eq. (40)
        h(:,j,userInd) = sqrtm(R(:,:,j,userInd))*v(:,j,userInd);
    end
end
if nbrOfRealizations == 1
    h_corr = zeros(M,UserNum);
    for user_n=1:UserNum
        h_corr(:,user_n) = h(:,:,user_n);
    end
else
    h_corr = h;
end
end
function points = generatePointsHexagon(nbrOfPoints,radius,minDistance)
%The hexagon 六角形的 is divided into three rhombus. Each point is uniformly
%distributed among these rhombus 菱形；斜方形； %
whichRhombus = randi(3,nbrOfPoints);
%Place the points uniformly distributed in a square
xDim = radius*rand(nbrOfPoints);
yDim = radius*rand(nbrOfPoints);
%Rotate the points in the squares to make them uniformly distributed in
%the right rhombus
points = -1i*xDim + (sqrt(3)/2+1i/2)*yDim;
points = points .* exp(1i*whichRhombus*2*pi/3);
%Remove points that are closer to the origin than "minDistance" and
%generate new points in the hexagon to replace these ones.
if nargin>2
    notReady = abs(points) < minDistance;
    if ~isempty(notReady)    
        points(notReady) = generatePointsHexagon([sum(notReady(:)) 1],radius,minDistance);    
    end
end
end
%% 画图
function  plot_snr_bler(SNR_Matrix,BLER_Matrix)
% LineStyles='-bs -kv -ro -cs --bs --kv --ro --cs --md -gv -md -rp --bs --gv --rp --ko -m< -yd --bs -c> --gv --rp --co --m< --kd --y>';
 LineStyles='-bs --bs -gd -gd -kv --kv -ro --ro -b+ --b+ -g* -g* -kd --kd -rp --rp -cs --cs -mv --mv -md -rp --bs --gv --rp --ko -m< -yd --bs -c> --gv --rp --co --m< --kd --y>';
Rows=size(BLER_Matrix,1);
LineStyles=parse(LineStyles);
MarkerSize =6;
LineWidth = 1;
for i=1:Rows
    plot(SNR_Matrix,BLER_Matrix(i,:),LineStyles(i,:),'LineWidth',LineWidth,'MarkerSize',MarkerSize);
    hold on;
    grid on;
end
axis([min(min(SNR_Matrix)) max(max(SNR_Matrix)) min(min(BLER_Matrix)) max(max(BLER_Matrix))  ]);
end
function [x] = parse(inStr) %LineStyles=parse(LineStyles);
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
function myboldify(h,MarkerSize,YLabelFontSize,FontSize,LineWidth,LegendFontSize,TitleFontSize)
%h  =gcf; % 获得当前figure 句柄，大家需要用这个模板来画图，仔细调整写出的文章才
%FontSize= 24 ; LineWidth = 3;TitleFontSize = 20; LegendFontSize = 36; axis_ratio=1.5; %myboldify(h,FontSize,LineWidth,LegendFontSize,TitleFontSize);
%  myboldify(h,FontSize,LineWidth,LegendFontSize,TitleFontSize)

% myboldify: make lines and text bold 
% myboldify boldifies the current figure
% myboldify(h) applies to the figure with the handle h
if nargin < 1
    h = gcf; 
    MarkerSize=9; YLabelFontSize =24; FontSize= 24 ; 
    LineWidth = 2;TitleFontSize =24; LegendFontSize =36; axis_ratio=1.5; %myboldify(h,FontSize,LineWidth,LegendFontSize,TitleFontSize);
end
ha = get(h, 'Children'); % the handle of each axis
for i = 1:length(ha)
    
    if strcmp(get(ha(i),'Type'), 'axes') % axis format
        set(ha(i), 'FontSize', LegendFontSize);      % tick mark and frame format
        set(ha(i), 'LineWidth', LineWidth);

        set(get(ha(i),'XLabel'), 'FontSize', YLabelFontSize);
        %set(get(ha(i),'XLabel'), 'VerticalAlignment', 'top');

        set(get(ha(i),'YLabel'), 'FontSize', YLabelFontSize);
        %set(get(ha(i),'YLabel'), 'VerticalAlignment', 'baseline');

        set(get(ha(i),'ZLabel'), 'FontSize', FontSize);
        %set(get(ha(i),'ZLabel'), 'VerticalAlignment', 'baseline');

        set(get(ha(i),'Title'), 'FontSize', TitleFontSize);
        %set(get(ha(i),'Title'), 'FontWeight', 'Bold');
    end
    
    hc = get(ha(i), 'Children'); % the objects within an axis
    for j = 1:length(hc)
        chtype = get(hc(j), 'Type');
        if strcmp(chtype(1:4), 'text')
            set(hc(j), 'FontSize', LegendFontSize); % 14 pt descriptive labels
        elseif strcmp(chtype(1:4), 'line')
            set(hc(j), 'LineWidth', LineWidth);
            set(hc(j), 'MarkerSize', MarkerSize);
        elseif strcmp(chtype, 'hggroup')
            hcc = get(hc(j), 'Children');
            if strcmp(get(hcc, 'Type'), 'hggroup')
                hcc = get(hcc, 'Children');
            end
            for k = 1:length(hcc) % all elements are 'line'
                set(hcc(k), 'LineWidth', LineWidth);
                set(hcc(k), 'MarkerSize', LegendFontSize);
            end            
        end
    end
    
end
end
%% 保存文件
function Save_link=Save_link_file(readFile,writeFile)
fid = fopen(readFile);
fid_write = fopen(writeFile,'w');
while 1
     tline = fgetl(fid);
    if ~ischar(tline)
        break, 
    end
    %tline = fgetl(fid);
    fprintf(fid_write,'%s\n',tline);
    %disp(tline)
end
fclose(fid);
fclose(fid_write);
Save_link=1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
