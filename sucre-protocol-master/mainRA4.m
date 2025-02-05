%There are three Random Access Protocol that are considered:
%1: SUCRe
%2: ACBPC
%3: Proposed
% [meanWaitingTime,finalWaitingTimes]=ACPC_Function(RandomAccess,nbrOfRAblocks);
clc;clear all; close all;
%实际上是 在一个码里面选择
L = 12; K = 10; Q =(L *(L-1))/2; (1-1/L)^(K-1)
(1-1/Q)^(K-1)

filenameTmp ='test_L';%Method_1_3_shadowFadingStddB factor_ratio_userNum 可将仿真参数作为文件名
mkdir_str=strcat('.\Simulation\',filenameTmp);
mkdir(mkdir_str);%一运行就会在当前文件夹下创建simulation文件夹
mkdir_str1 =strcat(mkdir_str,'\'); mkdir_str =strcat(mkdir_str1,filenameTmp); mkdir_str =strcat(mkdir_str,'.m');
Save_link1=Save_link_file('mainRA4.m',mkdir_str); %将当前
SUCRe =1; K0values = [100 200 300 500 1000:2000:13000]; %Range of number of inactive UEs in the cell 100 200 300 500 1000:5000:21000
% K0values = [100  1000:4000:18000];
K0values = [100:100:900 1000:2000:21000];
RandomSeed = 0; Kneighboringcells = 10;  SumIndex =0;
Default = 0;
Chan = 1; % 1:Uncorr Rayleigh   2:Corr Rayleigh   3:Los   
disInterval = 15; % distance interval
L_range = [ 2 3]; % 时隙数目
method_range =[1]; % 1:No Inter.  2: Inter.
nbrOfRAblocks_range =[200];   M_Appro_range = [1];   M_range = [100:50:100];   cellRadius_range =[250:500:250]; 
pA_range = [1:1]/10;  taup_range = [6:5:6]; Intra_Cell_Ratio_range =[10]/10;
PL_exp_range =[38:4:38]/1;   Kneighboringcells_range =[10:5:10];   EdgeRatio_range =[100:10:100]/100;    factor_range = [100:10:100]/100;
CompRatio_range =[10:4:10]/10;   EdgeCompRatio_range =[0 ]/10;   biasDelta_range = [0:0.5:0];   shadowFadingStddB_nlos_range = [8:8];  shadowFadingStddB_los_range = [4:4]; RandomAccess_range =[1];
for L = L_range for nbrOfRAblocks = nbrOfRAblocks_range for M_Appro = M_Appro_range for M = M_range   for taup = taup_range
          for cellRadius = cellRadius_range;  for pA =   pA_range    for Intra_Cell_Ratio =Intra_Cell_Ratio_range    for PL_exp = PL_exp_range   for Kneighboringcells =Kneighboringcells_range  for EdgeRatio = EdgeRatio_range   for factor =  factor_range    for CompRatio = CompRatio_range    for EdgeCompRatio = EdgeCompRatio_range
                                            for biasDelta = biasDelta_range    for shadowFadingStddB = shadowFadingStddB_nlos_range   for shadowFadingStddB_los = shadowFadingStddB_los_range    for RandomAccess =RandomAccess_range
                                                    if RandomSeed ==0 rand('state',12345);  randn('state',12345*3);  end% 保证每次SNR循环，初始种子都一样
  %[K0values,finalWaitingTimesACBPC,meanWaitingTimeACBPC,meanWaitingTimeBaseline,finalWaitingTimesBaseline,meanCollidingUsersNumACBPC] = ACPC_Function(RandomAccess,Para,nbrOfRAblocks);
Para.K0values =K0values;  Para.pA =pA ; Para.taup =taup ; Para.EdgeRatio =EdgeRatio ;  Para.factor =factor ;
Para.PL_exp =PL_exp;%;  path loss exponent
Para.method_range =method_range;
Para.L =L;
Para.Kneighboringcells =Kneighboringcells;  Para.shadowFadingStddB = shadowFadingStddB;  Para.shadowFadingStddB_los = shadowFadingStddB_los;
Para.cellRadius =cellRadius ; Para.Default =Default; Para.M_Appro =M_Appro;
Para.M =M ;     Para.cellRadius = cellRadius;    Para.CompRatio = CompRatio;   Para.EdgeCompRatio = EdgeCompRatio; Para.biasDelta = biasDelta;
Para.Intra_Cell_Ratio = Intra_Cell_Ratio;
Para.Chan = Chan;    Para.disInterval = disInterval;
[K0values,DataAll,DisPower,Disrange]=ACPC_Function(RandomAccess,Para,nbrOfRAblocks);
SumIndex = SumIndex+1;
if SumIndex ==1
    DataMatrix =DataAll;
    PowerMatrix = DisPower;
else
    DataMatrix(size(DataMatrix,1)+1: size(DataMatrix,1)+size(DataAll,1),:) =DataAll;
    PowerMatrix(size(PowerMatrix,1)+1: size(PowerMatrix,1)+size(DisPower,1),:) =DisPower;
end
strsave= strcat('.\Simulation\',filenameTmp,'\');
strsave= strcat(strsave,filenameTmp); s=['save ' strsave]; eval(s);% 保持.mat 文件，以后仿真结果可以再次确认,以后一定注意可以再次画图。
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

%% plot
DataMatrix
figure
hold on; box on;
plot_snr_bler(K0values,DataMatrix)
set(gca,'Fontname','Monospaced'); set(gca,'FontSize',12);
xlabel('Number of Inactive UEs'); ylabel('Average Number of Access Attempts');
legend('L=1','L=2','L=3','Location','SouthEast');ylim([0 10]);
stop = 1;
%%
%15600.00-4838.31+30

dd=[8600.00	
30000.00 
36056.02
20000.00
7416.00
9655.77
]
sum(dd)
plotFig =0;DataMatrix
if plotFig ==1
    filenameTmp ='test_factor1';%可将仿真参数作为文件名
    mkdir_str=strcat('.\Simulation\',filenameTmp);
    mkdir(mkdir_str);%一运行就会在当前文件夹下创建simulation文件夹
    mkdir_str1 =strcat(mkdir_str,'\');
    mkdir_str =strcat(mkdir_str1,filenameTmp);
    mkdir_str =strcat(mkdir_str,'.m');
    strsave= strcat('.\Simulation\',filenameTmp,'\');
    strsave= strcat(strsave,filenameTmp); s=['load ' strsave]; eval(s);% 保持.mat 文件，以后仿真结果可以再次确认,以后一定注意可以再次画图。
    close all;
    MarkerSize=9; LineWidth =2; LineMethod =1; PlotMethod =0; FontSize=22;
    % DataMatrix(end+1,:) =meanWaitingTimeBaseline.';
    SNR_Matrix =repmat(K0values,size(DataMatrix,1),1);
    SubCaseNum = 1*1*3;%size(DataMatrix,1) ;    %size(DataMatrix,1);
    PlotRatio = 1*1; StepNum = 1; startIndex =1+3*0;
    NumofFiles =SubCaseNum;
    for i=1:size(DataMatrix,1)/SubCaseNum/PlotRatio;
        figure;
        DataMatrix_tmp =DataMatrix((i-1)*SubCaseNum+1:i*SubCaseNum,:);
        SNR_Matrix_tmp =SNR_Matrix((i-1)*SubCaseNum+1:i*SubCaseNum,:);
        selectedIndex =[startIndex:StepNum:SubCaseNum]
        DataMatrix_tmp =DataMatrix_tmp(selectedIndex,:);
        SNR_Matrix_tmp =SNR_Matrix_tmp(selectedIndex,:);
        NumofFiles =size(SNR_Matrix_tmp,1);
        Plot_SIC_results=Plot_SIC(MarkerSize,LineWidth,FontSize,SNR_Matrix_tmp,DataMatrix_tmp,NumofFiles,LineMethod,PlotMethod)'
        %%Plot simulation results
        legend('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30');
    end
    
    
end


style = {'r-.+','b-.+','m-o','g-o','c-s', 'r-s','m-s','g-s','r-o','k--'};
idx=1;
figure
hold on; box on;
% plot(K0values,DataMatrix(2,:),'c-s','LineWidth',1)
for i=1:3:size(DataMatrix,1)
    plot(K0values,DataMatrix(i,:),style{idx},'LineWidth',1);
    idx = idx + 1; 
end
plot(K0values,DataMatrix(3,:),'k--','LineWidth',1)
ax1 = gca;
set(ax1,'FontSize',12);
xlabel('Number of Inactive UEs');
ylabel('Average Number of Access Attempts');
legend('Proposed: biasDelta=-2','Proposed: biasDelta=-1.5','Proposed: biasDelta=-1','Proposed: biasDelta=-0.5','Proposed: biasDelta=0','Proposed: biasDelta=0.5','Proposed: biasDelta=1','Proposed: biasDelta=1.5','Proposed: biasDelta=2','Baseline','Location','SouthEast');
% legend('SUCRe: No interf.','SUCRe: Interf.','Baseline','Proposed: No interf.','Proposed: Interf.','Location','SouthEast');
% legend('SUCRe: No interf.','SUCRe: Interf.','Baseline','ACBPC: No interf.','ACBPC: Interf.','Proposed: No interf.','Proposed: Interf.','Location','SouthEast');
title('Uncorr Rayleigh, \tau_p = 10, No Interf.');
ylim([0 10]);

figure
hold on; box on;
plot(Disrange, PowerMatrix(1,:), 'k--o', 'LineWidth',1);
plot(Disrange, PowerMatrix(2,:), 'b-.+', 'LineWidth',1);
% plot(Disrange, PowerMatrix(3,:), 'r-s', 'LineWidth',1);
xlabel('Distance to the BS [m]');
ylabel('Average transmit power per attempt/\rho^{SUCRe}');
legend('SUCRe','ACBPC','Proposed','Location','NorthWest');
title('Uncorr Rayleigh');
axis([0 250 min(min(PowerMatrix))*1.0 1.2]);
% subplot(122)
% hold on; box on;
% yyaxis left;
% plot(K0values,finalWaitingTimes(end,:,1)./sum(finalWaitingTimes(:,:,1),1),'r-.+','LineWidth',1);
% plot(K0values,finalWaitingTimes(end,:,2)./sum(finalWaitingTimes(:,:,2),1),'b-.+','LineWidth',1);
% plot(K0values,finalWaitingTimesBaseline(end,:)./sum(finalWaitingTimesBaseline,1),'k--','LineWidth',1);
% xlabel('Number of Inactive UEs');
% ylabel('Fraction of Failed Access Attempts');
%legend('ACBPC: No interf.','ACBPC: Interf.','Baseline','Location','SouthEast');

% yyaxis right;
% plot(K0values,meanCollidingUsersNum(:,1),'r--x','LineWidth',1);
% plot(K0values,meanCollidingUsersNum(:,2),'b--x','LineWidth',1);
% xlabel('Number of Inactive UEs');
% ylabel('Average Number of Colliding UEs');
% % legend('ACBPC: No interf.','ACBPC: Interf.','Baseline','ACBPC: No interf.','ACBPC: Interf.','Location','SouthEast');
% ylim([0 40])
function  [K0values,DataAll,DisPower,Disrange,meanWaitingTime,finalWaitingTimes,meanWaitingTimeBaseline]=ACPC_Function(RandomAccess,Para,nbrOfRAblocks)
%function  [K0values,finalWaitingTimesACBPC,meanWaitingTimeACBPC,meanWaitingTimeBaseline,finalWaitingTimesBaseline,meanCollidingUsersNumACBPC]=ACPC_Function(RandomAccess,nbrOfRAblocks)
%There are three Random Access Protocol that are considered:
%1: SUCRe; %2: ACBPC; %3: Proposed
Default = Para.Default;
K0values =Para.K0values;
pA = Para.pA;
taup =Para.taup;
factor =Para.factor;
Kneighboringcells =Para.Kneighboringcells;
cellRadius = Para.cellRadius; %Set cell radius (in meter)
M = Para.M;   %Number of BS antennas
M_Appro = Para.M_Appro;   %Number of BS antennas
shadowFadingStddB  = Para.shadowFadingStddB; % Standard deviation of shadow fading in non-line-of-sight
shadowFadingStddB_los = Para.shadowFadingStddB_los; % Standard deviation of shadow fading in line-of-sight
CompRatio  = Para.CompRatio;% compensation of nearby users
EdgeRatio = Para.EdgeRatio;%;
PL_exp = Para.PL_exp;%;  path loss exponent
Intra_Cell_Ratio = Para.Intra_Cell_Ratio;%;  path loss exponent
EdgeCompRatio = Para.EdgeCompRatio; % compensation of edge-users
biasDelta = Para.biasDelta; % bias term
method_range = Para.method_range; % 1:No Inter. 2: Inter. 
L = Para.L; % 时隙数目
Chan = Para.Chan; % channel models
disInterval = Para.disInterval; % distance interval
powerCounts = zeros(1,ceil((cellRadius-0.1*cellRadius)/disInterval));
transPower = zeros(1,ceil((cellRadius-0.1*cellRadius)/disInterval));
Disrange = zeros(1,ceil((cellRadius-0.1*cellRadius)/disInterval));
if  Default ==1
    M = 100;%Number of BS antennas
    Kneighboringcells = 10; %Set number of active UEs in the neighboring cells 设置相邻小区中活跃用户的数量
    K0values = [100 200 300  500 1000:3000:15000]; %Range of number of inactive UEs in the cell
    factor = 0.95; % 功率降低指数
    pA = 0.001; %Probability that an inactive UE wants to become active in a given block
    taup = 10; %Number of RA pilot signals
    cellRadius = 250; %Set cell radius (in meter)
    shadowFadingStddB = 10; %Standard deviation of shadow fading
    L_range = [1 2 3]; % 时隙数目
end

maxAttempts = 10; %Maximum number of attempts to send RA pilots before a UE gives up
%Probability of retransmitting an RA pilot in each follow RA block, when the first transmission was failed.
tryAgainProb = 0.5;
%%Define simulation scenario
q = 1; %Transmit power of the BS
sigma2 = 1; %Noise variance
rhoSUCRe = 1; %Transmit power of UEs in the cell
rho = 1;
rhoAverage = sigma2;
rhoSUCRedB = - 98.65 + 35.3 + PL_exp*log10(cellRadius);
rhoIntercell = 1; %Set transmit power of users in adjacent cells 设置相邻小区用户的传输功率
%Compute locations of all neighboring BSs 计算所有邻近BSs的位置
neighboringBSs = 2*cellRadius*sqrt(3)/2 * exp(1i*(0:5)*pi/3);
%Generate noise realizations at the BS in the center cell 中心小区基站的噪声
n = sqrt(1/2)*(randn(M,taup,nbrOfRAblocks)+1i*randn(M,taup,nbrOfRAblocks));

%Matrices to store the simulation results, in terms of waiting time for the
%RandomAccess protocol (without or with inter-cell interference) and baseline scheme 累积不同尝试次数的个数
finalWaitingTimes = zeros(maxAttempts+1,length(K0values),length(method_range));
finalWaitingTimesBaseline = zeros(maxAttempts+1,length(K0values));
NumberOfCollidingUsers = zeros(nbrOfRAblocks,length(K0values),2) ;
meanWaitingTime = zeros(length(K0values),length(method_range));
%Generate user locations in neighboring cells 在相邻小区中生成用户位置
userLocationsNeighboring = generatePointsHexagon([Kneighboringcells,nbrOfRAblocks,length(neighboringBSs)],cellRadius,0.1*cellRadius);
%Generate shadow fading realizations of users in neighboring cells, both
%within that cell and to the cell under study 生成邻近小区用户的阴影衰落，包括该小区的用户和被研究的小区
shadowFadingRealizationsWithinOwnCellUplink = randn([Kneighboringcells,nbrOfRAblocks,length(neighboringBSs)]);
shadowFadingRealizationsIntercellUplink = randn([Kneighboringcells,nbrOfRAblocks,length(neighboringBSs)]);
%Go through all users in neighboring cells
 for k = 1:Kneighboringcells
     notReady = 1;
     while sum(notReady)>0
         notReady = zeros(length(neighboringBSs),nbrOfRAblocks);
         for j = 1:length(neighboringBSs)
             %Check which of the users that are served by the right BS
             notReady(j,:) = (shadowFadingStddB*shadowFadingRealizationsWithinOwnCellUplink(k,:,j)...,
                 - PL_exp*log10(abs(userLocationsNeighboring(k,:,j))) < shadowFadingStddB*shadowFadingRealizationsIntercellUplink(k,:,j)...,
                 - PL_exp*log10(abs(userLocationsNeighboring(k,:,j)-neighboringBSs(j))) );
             %Make new random shadow fading realizations for those users in
             %the neighborin cell that have a better channel to the center BS
             shadowFadingRealizationsWithinOwnCellUplink(k,notReady(j,:)>0,j) = randn(1,sum(notReady(j,:)>0));
         end
     end
end
%Compute the total inter-cell interference in the uplink and downlink in each channel realization
interCellVarianceUplink = zeros(1,nbrOfRAblocks);
for j = 1:length(neighboringBSs)
    %Note:  -98.65 dBm represents the noise variance
    POWER = -( 98.65 - 35.3 - PL_exp*log10(abs(userLocationsNeighboring(:,:,j) + neighboringBSs(j))) + shadowFadingStddB*shadowFadingRealizationsIntercellUplink(:,:,j)  );
    if  RandomAccess==1
        interCellVarianceUplink = interCellVarianceUplink + sum(rhoIntercell*10.^( (rhoSUCRedB + 98.65 - 35.3 - PL_exp*log10(abs(userLocationsNeighboring(:,:,j) + neighboringBSs(j))) + shadowFadingStddB*shadowFadingRealizationsIntercellUplink(:,:,j)  )/10   ),1) ;
    end
    if  RandomAccess==2
        interCellVarianceUplink = interCellVarianceUplink + sum(rhoIntercell*10.^( (min(POWER,rhoSUCRedB)+ 98.65 - 35.3 - PL_exp*log10(abs(userLocationsNeighboring(:,:,j) + neighboringBSs(j))) + shadowFadingStddB*shadowFadingRealizationsIntercellUplink(:,:,j)  )/10   ),1) ;
        %interCellVarianceUplink = interCellVarianceUplink + sum(rhoIntercell*10.^( (rhoSUCRedB+ 98.65 - 35.3 - PL_exp*log10(abs(userLocationsNeighboring(:,:,j) + neighboringBSs(j))) + shadowFadingStddB*shadowFadingRealizationsIntercellUplink(:,:,j)  )/10   ),1) ;
    end
    if  RandomAccess==3
        index = find(10.^(POWER/10)>10.^(rhoSUCRedB/10));
        POWER(index) = rhoSUCRedB;
        lower_bound = 10.^(rhoSUCRedB/10)*EdgeRatio;  %min(POWER,rhoSUCRedB)
        %index = find(POWER>lower_bound);
        index_1 = find(10.^(POWER/10)>lower_bound);
        POWER(index_1) = 10*log10((10.^(POWER(index_1)/10))*factor);
        interCellVarianceUplink = interCellVarianceUplink + sum(rhoIntercell*10.^( (min(POWER,rhoSUCRedB) + 98.65 - 35.3 - PL_exp*log10(abs(userLocationsNeighboring(:,:,j) + neighboringBSs(j))) + shadowFadingStddB*shadowFadingRealizationsIntercellUplink(:,:,j)  )/10   ),1) ;
%         interCellVarianceUplink = interCellVarianceUplink + sum(rhoIntercell*10.^( (rhoSUCRedB + 98.65 - 35.3 - PL_exp*log10(abs(userLocationsNeighboring(:,:,j) + neighboringBSs(j))) + shadowFadingStddB*shadowFadingRealizationsIntercellUplink(:,:,j)  )/10   ),1) ;
    end
end
%Compute the average uplink inter-cell interference, which is used in the
%bias terms
interCellBias = mean(interCellVarianceUplink);
%Go through all different number of inactive UEs
for indProb = 1:length(K0values)
    disp(['K0 values: ' num2str(indProb) ' out of ' num2str(length(K0values))]); % Display simulation progress
    K0 = K0values(indProb);  % Extract current value of the number of inactive UEs
    %Generate the number of UEs that wish to access the network (for the first time) in each of the RA blocks
    newUsers = binornd(K0,pA,[nbrOfRAblocks 1]);
    %There are three methods that are considered:
    %1: RandomAccess without inter-cell interference
    %2: RandomAccess with inter-cell interference
    %3: Baseline scheme
    for method = method_range
        %Initiate the set of UEs that have failed to access the network
        waitingTime = []; %Contains number of access attempts
        waitingBetas = []; %Channel variance of the users
        waitingAngles = []; %Angles of the users
        waitingDistance = []; %Distance of the users
        waitingIntercellInter = []; %Downlink inter-cell inteference variance
        waitingUsers_Comp_index = [];
        waitingRhodB_Comp = [];
        %Set the inter-cell interference parameters
        if method == 2
            qNeighbor = taup*q; %Transmit power of neighbouring BSs
            rhoIntercell = rhoSUCRe; %Set transmit power of users in adjacent cells
        else
            qNeighbor = 0; %No transmit power of neighbouring BSs
            rhoIntercell = 0; %No transmit power of users in adjacent cells
        end
        %Go through all RA blocks that are considered in the Monte-Carlo simulations
        for r = 1:nbrOfRAblocks
            newUserLocations = generatePointsHexagon([newUsers(r) 1],cellRadius,0.1*cellRadius);   %Generate locations of new UEs that try to access the network
            newUserDistance = abs(newUserLocations);  %Extract user distances from the BS
            newUserAngles = angle(newUserLocations); %Extract angles between users and the BS
            newShadowFading = randn(newUsers(r),1);   %Generate shadow fading realizations
            shadowFadingRealizationsIntercellDownlink = randn(newUsers(r),length(neighboringBSs));   %Generate shadow fading realizations for downlink inter-cell interference
            %Go through all new users and make sure that they are always served by the BS in the own hexagonal cell
            %(even when shadow fading is taken into account)
            notReady = 1;
            
            %shadowFadingStddB_nlos = 10; shadowFadingStddB_los = 4;
            %  betas_nlos = 10.^( (27+ 98.65 - 34.53 - 38*log10(newUserDistance) + shadowFadingStddB_nlos*shadowFadingRealizationsIntercellDownlink)/10);
            % % Table 5.1. Environment parameters of  [2] or (5.2-3) below of  [2]
            % betas_los = 10.^( (27+ 98.65 - 30.18 - 26*log10(userDistance) + shadowFadingStddB_los*shadowFadingRealizations)/10);
            % %Generate uncorrelated Rayleigh fading channel realizations
            % h_iid = (randn(Mmax,nbrOfRealizations,maxUsers)+1i*randn(Mmax,nbrOfRealizations,maxUsers));
            % h_iid = repmat(sqrt(betas_nlos/2),[Mmax 1 1 ]) .* h_iid;
            %Define distance between antennas in the ULA, measured in wavelengths
            interAntennaDistance = 0.5;
            %Select correlation factor in the exponential correlation model
            correlationFactor = 0.7;
            %Generate line-of-sight channel realizations using Eq. (43)
            % h_los = exp( repmat((0:Mmax-1)',[1 nbrOfRealizations maxUsers]) .* repmat(-2i*pi*interAntennaDistance*sin(userAngles),[Mmax 1 1]) );
            % h_los = repmat(sqrt(betas_los),[Mmax 1 1 ]) .* h_los;
            % %Select correlation factor in the exponential correlation model
            % correlationFactor = 0.7;
            % %Vector to store correlated Rayleigh fading
            % h_corr = zeros(Mmax,nbrOfRealizations,maxUsers);
            while sum(notReady)>0
                notReady = zeros(newUsers(r),1);
                for j = 1:length(neighboringBSs)
                    %Check which of the users that are served by the right BS
                    notReady = notReady + (shadowFadingStddB*newShadowFading - PL_exp*log10(newUserDistance) < shadowFadingStddB*shadowFadingRealizationsIntercellDownlink(:,j) - PL_exp*log10(abs(newUserLocations-neighboringBSs(j))) );
                end
                %Make new random shadow fading realizations for those users that
                %have a better channel to the neighboring BS
                newShadowFading(notReady>0) = randn(sum(notReady>0),1);
                shadowFadingRealizationsIntercellDownlink(notReady>0,:) = randn(sum(notReady>0),length(neighboringBSs));
            end
            %Compute average signal gain for non-line-of-sight propagation
            %( -98.65 dBm represents the noise variance)
            % rhoSUCRedB = - 98.65 + 35.3 + PL_exp*log10(cellRadius) % THE FOURTH
            %betas_los = 10.^( (27+ 98.65 - 30.18 - 26*log10(userDistance) + shadowFadingStddB_los*shadowFadingRealizations)/10);
            rhoNewdB = - 98.65 + 35.3 + PL_exp*log10(newUserDistance)- shadowFadingStddB*newShadowFading;
            if  RandomAccess==1
                rhodB = rhoSUCRedB;
            end
            if  RandomAccess==2
                rhodB = min(rhoNewdB,rhoSUCRedB);
                %rhodB = rhoSUCRedB;
            end
            if  RandomAccess==4
                rhoNewdB_R = rhoNewdB;
                index = find(rhoNewdB_R>rhoSUCRedB); % &rhoNewdB>0)
                rhoNewdB_R(index) = rhoSUCRedB;
                lower_bound = 10*log10(10.^(rhoSUCRedB/10)*EdgeRatio);
                index = find(rhoNewdB_R>lower_bound); % &rhoNewdB>0)
                %rhoNewdB_R(index) = rhoNewdB_R(index).^factor; % 阈值以上合理降低功率
                rhoNewdB_R(index) = 10*log10((10.^(rhoNewdB_R(index)/10))*factor);
                newRhodB_Comp = rhoNewdB - rhoNewdB_R; % 记录降低的功率值（绝对值）
                rhodB = min(rhoNewdB_R,rhoSUCRedB);                
%                 rhodB = rhoSUCRedB;                
            end
            if  RandomAccess==3
                rhoNewdB_R = rhoNewdB;
                index = find(rhoNewdB_R>rhoSUCRedB); % &rhoNewdB>0)
                rhoNewdB_R(index) = rhoSUCRedB;
                shift_value = (rhoSUCRedB-rhoNewdB_R);
                shift_value = abs(shift_value).^factor.*sign(shift_value);
                rhoNewdB_R =rhoNewdB_R - 2*rand(size(shift_value,1),1).*shift_value; 
                
                lower_bound = 10*log10(10.^(rhoSUCRedB/10)*EdgeRatio);
                index = find(rhoNewdB_R>lower_bound); % &rhoNewdB>0)
                %rhoNewdB_R(index) = rhoNewdB_R(index).^factor; % 阈值以上合理降低功率
                rhoNewdB_R(index) = 10*log10((10.^(rhoNewdB_R(index)/10))*factor);
                newRhodB_Comp = rhoNewdB - rhoNewdB_R; % 记录降低的功率值（绝对值）
                rhodB = min(rhoNewdB_R,rhoSUCRedB);                
%                 rhodB = rhoSUCRedB;                
            end
            
            
            if Chan == 1 || Chan == 2
                %newBetas = 10.^((rhodB+ 98.65 - 35.3 - PL_exp*log10(newUserDistance) + shadowFadingStddB*newShadowFading  )/10   );
               % newShadowFading = newShadowFading+1/sqrt(2)*randn(size(newShadowFading,1),1); %  20210106;
                newBetas = 10.^((rhodB+ 98.65 - 35.3 - PL_exp*log10(newUserDistance) + shadowFadingStddB*newShadowFading  )/10   );                
            else
                newBetas = 10.^((rhodB+ 98.65 - 30.18 - 26*log10(newUserDistance) + shadowFadingStddB_los*newShadowFading  )/10   );
            end
%             newBetas = 10.^((rhodB+ 98.65 - 35.3 - PL_exp*log10(newUserDistance) + shadowFadingStddB*newShadowFading  )/10   );
            %Compute the total inter-cell interference in the uplink and downlink in
            %each channel realization
            newIntercellVarianceDownlink = zeros(newUsers(r),1);
            for j = 1:length(neighboringBSs)
                %Note: -98.65 dBm represents the noise variance
                %POWER = -( 98.65 - 35.3 - PL_exp*log10(abs(newUserLocations-neighboringBSs(j))) + shadowFadingStddB*shadowFadingRealizationsIntercellDownlink(:,j)  );
                %if  RandomAccess==1
                newIntercellVarianceDownlink = newIntercellVarianceDownlink + qNeighbor*10.^( ( rhoSUCRedB + 98.65 - 35.3 - PL_exp*log10(abs(newUserLocations-neighboringBSs(j))) + shadowFadingStddB*shadowFadingRealizationsIntercellDownlink(:,j)  )/10   );
                %   end
                %   if  RandomAccess==2
                %       newIntercellVarianceDownlink = newIntercellVarianceDownlink + qNeighbor*10.^( ( min(POWER,rhoSUCRedB) + 98.65 - 35.3 - PL_exp*log10(abs(newUserLocations-neighboringBSs(j))) + shadowFadingStddB*shadowFadingRealizationsIntercellDownlink(:,j)  )/10   );
                %   end
                %   if  RandomAccess==3
                %       %lower_bound = 20;
                %       index = find(POWER>rhoSUCRedB); % &rhoNewdB>0)
                %       POWER(index) = rhoSUCRedB;
                %       index = find(POWER>lower_bound);
                %       POWER(index) = POWER(index).^factor;
                %       newIntercellVarianceDownlink = newIntercellVarianceDownlink + qNeighbor*10.^( (min(POWER,rhoSUCRedB) + 98.65 - 35.3 - PL_exp*log10(abs(newUserLocations-neighboringBSs(j))) + shadowFadingStddB*shadowFadingRealizationsIntercellDownlink(:,j)  )/10   );
                %   end
            end
            interCellVarianceDownlink = [newIntercellVarianceDownlink; waitingIntercellInter];
            %      Users_Comp_index = [newUsers_Comp_index ;waitingUsers_Comp_index];
            if  or(RandomAccess==3,RandomAccess==4)
                rhodB_Comp = [newRhodB_Comp; waitingRhodB_Comp]; % 组合新用户和未成功接入用户降低的功率值
            end
            %Combine the new UEs with the ones that have made previous access attempts
            betas = [newBetas; waitingBetas];
            UserAngles = [newUserAngles; waitingAngles]; % 组合新用户和未成功接入用户的角度
            UserDistance = [newUserDistance; waitingDistance]; % 组合新用户和未成功接入用户的距离
            numberOfAccessingUsers = length(betas);  %Compute number of UEs that will send pilots
            %Randomize if each of the UEs that retransmit pilots should
            %really send a pilot in this RA block. One means retransmit and
            %zero means do not retransmit in this block
            shouldWaitingUsersRetransmit = binornd(1,tryAgainProb,size(waitingTime));
            %Create a list of the UEs that will send pilots (all new UEs
            %transmit pilots)
            accessAttempt = [ones(newUsers(r),1) ; shouldWaitingUsersRetransmit]; %新用户+失败用户
            %Randomize which of the pilots that each of the UEs are using
            pilotSelections = zeros([numberOfAccessingUsers L]);
            successfulAccess = zeros([numberOfAccessingUsers L]);% 记录每一个时隙用户是否重传
            for index_numberOfAccessingUsers = 1:numberOfAccessingUsers
                pilotSelections(index_numberOfAccessingUsers,:) = randperm(taup,L);% 选择
                %tmp =randperm(taup,L)；pilotSelections(index_numberOfAccessingUsers,:) = tmp(1)*ones(1,L;)
                %     Pilot_user(index_numberOfAccessingUsers,:)=randi([1,Pilot_N],1,Pilot_n);
            end
            pilotSelections = accessAttempt.*pilotSelections;
            %Count the number of pilots that each of the UEs will have
            %transmitted, after this block
            accessAttempts = [ones(newUsers(r),1); waitingTime+shouldWaitingUsersRetransmit];  %重传的次数
            %Check if there is at least on UE that transmit pilots in this RA block
            if ~isempty(accessAttempts)
                %Generate channel realizations
                if Chan == 1 %Uncorrelated Rayleigh fading  
                    h = (randn(M,numberOfAccessingUsers)+1i*randn(M,numberOfAccessingUsers));
                    h = repmat(sqrt(betas'/2),[M 1]) .* h;
                elseif Chan == 2 %Correlated Rayleigh fading
                    h_iid = (randn(M,numberOfAccessingUsers)+1i*randn(M,numberOfAccessingUsers));
                    h_iid = repmat(sqrt(betas'/2),[M 1]) .* h_iid;
                    for userInd = 1:numberOfAccessingUsers
                        R = toeplitz((correlationFactor*exp(1i*sin(UserAngles(userInd)))).^(0:M-1));
                        h(:,userInd) = sqrtm(R)*h_iid(:,userInd);
                    end
                elseif Chan == 3 %Line-of-sight with full power
                    h = exp( repmat((0:M-1)',[1 numberOfAccessingUsers]) .* repmat(-2i*pi*interAntennaDistance*sin(UserAngles'),[M 1]) );
                    h = repmat(sqrt(betas'),[M 1]) .* h;
                end
                %Generate noise plus inter-cell interference realizations
                %at the UEs.
                if method == 1
                    noiseInterfvariance = sigma2;
                    eta = sqrt(sigma2/2)*(randn(1,numberOfAccessingUsers)+1i*randn(1,numberOfAccessingUsers));
                elseif method == 2
                    noiseInterfvariance = sigma2+interCellVarianceUplink(1,r);
                    eta = sqrt((sigma2+interCellVarianceDownlink')/2).*(randn(1,numberOfAccessingUsers)+1i*randn(1,numberOfAccessingUsers));
                end
                %Prepare a list of UEs that succeed in the random access
                successfulAttempt = false(size(betas));
                
                if L == 1 || L == 2 ||  L == 3
                    %Go through all RA pilots
                    for t_1 = 1:taup
                        userIndices_1 = find(pilotSelections(:,1)==t_1);% 第一个时隙碰撞用户
                        userIndices = userIndices_1;
                        %Consider the SUCRe protocol
                        if method == 1 || method == 2
                            %Compute the received signal in Eq. (6)
                            yt = sqrt(taup*rho) * sum(h(:,userIndices),2) + sqrt(noiseInterfvariance)*n(:,t_1,r);
                            %Compute the precoding vector used by the BS
                            v = sqrt(q)*yt/norm(yt);
                            %Prepare a list of UEs that decide to retransmit pilot t
                            retransmit_1 = false(length(userIndices),1);
                            contendersEst_1 = zeros(length(userIndices),1);
                            %                         retransmit_matrix = zeros([numberOfAccessingUsers 2]);
                            %Go through the UEs that transmitted pilot t
                            for k = 1:length(userIndices)
                                z = sqrt(taup)*sum(conj(h(:,userIndices(k))).*v,1) + eta(1,userIndices(k));
                                if M_Appro==0
                                alphaEst_approx2 = exp(gammaln(M+1/2)-gammaln(M))^2*q*taup^2*rho*(betas(userIndices(k))./real(z)).^2-sigma2;
                                end
                                 if M_Appro==1
                                alphaEst_approx2 = M*q*taup^2*rho*(betas(userIndices(k))./real(z)).^2-sigma2;
                                 end  
%                                 alphaEst_approx2 = exp(gammaln(M+1/2)-gammaln(M))^2*q*taup^2*rho*(betas(userIndices(k))./real(z)).^2-sigma2;
                                if alphaEst_approx2<rho*betas(userIndices(k))*taup
                                    alphaEst_approx2 = rho*betas(userIndices(k))*taup;
                                end
                                if  RandomAccess==1
                                    if method == 1
                                        retransmit_1(k) =  rho*betas(userIndices(k))*taup>alphaEst_approx2/2-betas(userIndices(k))/sqrt(M);
                                    elseif method == 2
                                        retransmit_1(k) =  rho*betas(userIndices(k))*taup>(alphaEst_approx2-interCellBias)/2-betas(userIndices(k))/sqrt(M);
                                    end
                                end
                                if  RandomAccess==2
                                    randomNum = rand(1);
                                    if method == 1
                                        %retransmit(k) =  rho(userIndices(k))*betas(userIndices(k))*taup>alphaEst_approx2/2-betas(userIndices(k))/sqrt(M);
                                        contendersEst_1(k) =  round(alphaEst_approx2/(rhoAverage*taup)); % Compute estimate of contenders at user k
                                        retransmit_1(k) = randomNum<(1/contendersEst_1(k));                                        
                                    elseif method == 2
                                        %retransmit(k) =  rho(userIndices(k))*betas(userIndices(k))*taup>(alphaEst_approx2-interCellBias)/2-betas(userIndices(k))/sqrt(M);
                                        contendersEst_1(k) =  round((alphaEst_approx2-interCellBias)/(rhoAverage*taup)); % Compute estimate of contenders at user k
                                        retransmit_1(k) = randomNum<(1/contendersEst_1(k));                                        
                                    end
                                end
                                if or(RandomAccess==3,RandomAccess==4)
                                    %     ComValue = 4*rand(1);
                                    if method == 1
                                        %         SUCRe
                                        %         retransmit(k) =  rho*betas(userIndices(k))*taup...,
                                        %             >alphaEst_approx2/2-betas(userIndices(k))/sqrt(M);
                                        %         补偿因子CompRatio
                                        %         if UserDistance(userIndices(k)) < 1.1*cellRadius/2
                                        %
                                        %             if method == 1
                                        %                 retransmit(k) =  rho*betas(userIndices(k))*taup...,
                                        %                     >alphaEst_approx2/2-betas(userIndices(k))/sqrt(M);
                                        %             elseif method == 2
                                        %                 retransmit(k) =  rho*betas(userIndices(k))*taup...,
                                        %                     >(alphaEst_approx2-interCellBias)/2-betas(userIndices(k))/sqrt(M);
                                        %             end
                                        if betas(userIndices(k)) >0.99
                                            % retransmit(k) =  CompRatio*rho*betas(userIndices(k))*taup*10^(rhodB_Comp(userIndices(k))/10)...,
                                            retransmit_1(k) =  CompRatio*rho*betas(userIndices(k))*taup...,
                                                >alphaEst_approx2/2+biasDelta*betas(userIndices(k))/sqrt(M);
                                        else
                                            %  retransmit(k) =  rho*betas(userIndices(k))*taup*EdgeCompRatio*10^(rhodB_Comp(userIndices(k))/10)...,
                                            %  >alphaEst_approx2/2+biasDelta*betas(userIndices(k))/sqrt(M);
                                            retransmit_1(k) =  rho*betas(userIndices(k))*taup*10^(EdgeCompRatio*rhodB_Comp(userIndices(k))/10)...,
                                                >alphaEst_approx2/2+biasDelta*betas(userIndices(k))/sqrt(M);
                                        end
                                        %         原算法
                                        %         retransmit(k) =  rho*betas(userIndices(k))*taup*10^(CompRatio*rhodB_Comp(userIndices(k))/10)...,
                                        %             >alphaEst_approx2/2-betas(userIndices(k))/sqrt(M);
                                    elseif method == 2 % last fourth lines of Section IV.D
                                        %         SUCRe
                                        %         retransmit(k) =  rho*betas(userIndices(k))*taup...,
                                        %             >(alphaEst_approx2-Intra_Cell_Ratio*interCellBias)/2-betas(userIndices(k))/sqrt(M);
                                        %         补偿因子CompRatio
                                        %         if UserDistance(userIndices(k)) < 1.1*cellRadius/2
                                        if betas(userIndices(k)) >0.99
                                            %             retransmit(k) =  CompRatio*rho*betas(userIndices(k))*taup*10^(rhodB_Comp(userIndices(k))/10)...,
                                            retransmit_1(k) =  CompRatio*rho*betas(userIndices(k))*taup...,
                                                >(alphaEst_approx2-Intra_Cell_Ratio*interCellBias)/2+biasDelta*betas(userIndices(k))/sqrt(M);
                                        else
                                            retransmit_1(k) =  rho*betas(userIndices(k))*taup*10^(EdgeCompRatio*rhodB_Comp(userIndices(k))/10)...,
                                                >(alphaEst_approx2-Intra_Cell_Ratio*interCellBias)/2+biasDelta*betas(userIndices(k))/sqrt(M);
                                        end
                                        % 原算法
                                        % retransmit(k) =  rho*betas(userIndices(k))*taup*10^(CompRatio*rhodB_Comp(userIndices(k))/10)...,
                                        %   >(alphaEst_approx2-Intra_Cell_Ratio*interCellBias)/2-betas(userIndices(k))/sqrt(M);
                                        
                                    end
                                end
                            end
                            if sum(retransmit_1) == 1
                                successfulAccess(userIndices,1)=retransmit_1;
                                stop = 2;
                            end
                        end
                        %                     if method == 3
                        %                         if length(userIndices) == 1
                        %                             successfulAttempt(userIndices) = true;
                        %                             finalWaitingTimesBaseline(accessAttempts(userIndices),indProb) = finalWaitingTimesBaseline(accessAttempts(userIndices),indProb) + 1;
                        %                         end
                        %                     end
                    end
                end
                if L == 2 ||  L == 3
                    %Go through all RA pilots
                    for t_2 = 1:taup
                        userIndices_2 = find(pilotSelections(:,2)==t_2);
                        userIndices = userIndices_2;
                        if method == 1 || method == 2
                            yt = sqrt(taup*rho) * sum(h(:,userIndices),2) + sqrt(noiseInterfvariance)*n(:,t_2,r);
                            v = sqrt(q)*yt/norm(yt);
                            retransmit_2 = false(length(userIndices),1);
                            contendersEst_2 = zeros(length(userIndices),1);
                            %Go through the UEs that transmitted pilot t
                            for k = 1:length(userIndices)
                                z = sqrt(taup)*sum(conj(h(:,userIndices(k))).*v,1) + eta(1,userIndices(k));
                                if M_Appro==0
                                alphaEst_approx2 = exp(gammaln(M+1/2)-gammaln(M))^2*q*taup^2*rho*(betas(userIndices(k))./real(z)).^2-sigma2;
                                end
                                 if M_Appro==1
                                alphaEst_approx2 = M*q*taup^2*rho*(betas(userIndices(k))./real(z)).^2-sigma2;
                                 end  
%                                 alphaEst_approx2 = exp(gammaln(M+1/2)-gammaln(M))^2*q*taup^2*rho*(betas(userIndices(k))./real(z)).^2-sigma2;
                                if alphaEst_approx2<rho*betas(userIndices(k))*taup
                                    alphaEst_approx2 = rho*betas(userIndices(k))*taup;
                                end
                                if  RandomAccess==1
                                    if method == 1
                                        retransmit_2(k) =  rho*betas(userIndices(k))*taup>alphaEst_approx2/2-betas(userIndices(k))/sqrt(M);
                                    elseif method == 2
                                        retransmit_2(k) =  rho*betas(userIndices(k))*taup>(alphaEst_approx2-interCellBias)/2-betas(userIndices(k))/sqrt(M);
                                    end
                                end
                                if  RandomAccess==2
                                    randomNum = rand(1);
                                    if method == 1
                                        %retransmit(k) =  rho(userIndices(k))*betas(userIndices(k))*taup>alphaEst_approx2/2-betas(userIndices(k))/sqrt(M);
                                        contendersEst_2(k) =  round(alphaEst_approx2/(rhoAverage*taup)); % Compute estimate of contenders at user k
                                        retransmit_2(k) = randomNum<(1/contendersEst_2(k));
                                    elseif method == 2
                                        %retransmit(k) =  rho(userIndices(k))*betas(userIndices(k))*taup>(alphaEst_approx2-interCellBias)/2-betas(userIndices(k))/sqrt(M);
                                        contendersEst_2(k) =  round((alphaEst_approx2-interCellBias)/(rhoAverage*taup)); % Compute estimate of contenders at user k
                                        retransmit_2(k) = randomNum<(1/contendersEst_2(k));
                                    end
                                end
                                if or(RandomAccess==3,RandomAccess==4)
                                    %     ComValue = 4*rand(1);
                                    if method == 1
                                        %         SUCRe
                                        %         retransmit(k) =  rho*betas(userIndices(k))*taup...,
                                        %             >alphaEst_approx2/2-betas(userIndices(k))/sqrt(M);
                                        %         补偿因子CompRatio
                                        %         if UserDistance(userIndices(k)) < 1.1*cellRadius/2
                                        %
                                        %             if method == 1
                                        %                 retransmit(k) =  rho*betas(userIndices(k))*taup...,
                                        %                     >alphaEst_approx2/2-betas(userIndices(k))/sqrt(M);
                                        %             elseif method == 2
                                        %                 retransmit(k) =  rho*betas(userIndices(k))*taup...,
                                        %                     >(alphaEst_approx2-interCellBias)/2-betas(userIndices(k))/sqrt(M);
                                        %             end
                                        if betas(userIndices(k)) >0.99
                                            % retransmit(k) =  CompRatio*rho*betas(userIndices(k))*taup*10^(rhodB_Comp(userIndices(k))/10)...,
                                            retransmit_2(k) =  CompRatio*rho*betas(userIndices(k))*taup...,
                                                >alphaEst_approx2/2+biasDelta*betas(userIndices(k))/sqrt(M);
                                        else
                                            %  retransmit(k) =  rho*betas(userIndices(k))*taup*EdgeCompRatio*10^(rhodB_Comp(userIndices(k))/10)...,
                                            %  >alphaEst_approx2/2+biasDelta*betas(userIndices(k))/sqrt(M);
                                            retransmit_2(k) =  rho*betas(userIndices(k))*taup*10^(EdgeCompRatio*rhodB_Comp(userIndices(k))/10)...,
                                                >alphaEst_approx2/2+biasDelta*betas(userIndices(k))/sqrt(M);
                                        end
                                        %         原算法
                                        %         retransmit(k) =  rho*betas(userIndices(k))*taup*10^(CompRatio*rhodB_Comp(userIndices(k))/10)...,
                                        %             >alphaEst_approx2/2-betas(userIndices(k))/sqrt(M);
                                    elseif method == 2 % last fourth lines of Section IV.D
                                        %         SUCRe
                                        %         retransmit(k) =  rho*betas(userIndices(k))*taup...,
                                        %             >(alphaEst_approx2-Intra_Cell_Ratio*interCellBias)/2-betas(userIndices(k))/sqrt(M);
                                        %         补偿因子CompRatio
                                        %         if UserDistance(userIndices(k)) < 1.1*cellRadius/2
                                        if betas(userIndices(k)) >0.99
                                            %             retransmit(k) =  CompRatio*rho*betas(userIndices(k))*taup*10^(rhodB_Comp(userIndices(k))/10)...,
                                            retransmit_2(k) =  CompRatio*rho*betas(userIndices(k))*taup...,
                                                >(alphaEst_approx2-Intra_Cell_Ratio*interCellBias)/2+biasDelta*betas(userIndices(k))/sqrt(M);
                                        else
                                            retransmit_2(k) =  rho*betas(userIndices(k))*taup*10^(EdgeCompRatio*rhodB_Comp(userIndices(k))/10)...,
                                                >(alphaEst_approx2-Intra_Cell_Ratio*interCellBias)/2+biasDelta*betas(userIndices(k))/sqrt(M);
                                        end
                                        % 原算法
                                        % retransmit(k) =  rho*betas(userIndices(k))*taup*10^(CompRatio*rhodB_Comp(userIndices(k))/10)...,
                                        %   >(alphaEst_approx2-Intra_Cell_Ratio*interCellBias)/2-betas(userIndices(k))/sqrt(M);
                                        
                                    end
                                end
                                stop =1;
                            end
                            if sum(retransmit_2) == 1
                                successfulAccess(userIndices,2)=retransmit_2;
                            end
                        end
                    end
                end
                if L == 3
                    %Go through all RA pilots
                    for t_3 = 1:taup
                        userIndices_3 = find(pilotSelections(:,3)==t_3);
                        userIndices = userIndices_3;
                        if method == 1 || method == 2
                            yt = sqrt(taup*rho) * sum(h(:,userIndices),2) + sqrt(noiseInterfvariance)*n(:,t_3,r);
                            v = sqrt(q)*yt/norm(yt);
                            retransmit_3 = false(length(userIndices),1);
                            contendersEst_3 = zeros(length(userIndices),1);
                            %Go through the UEs that transmitted pilot t
                            for k = 1:length(userIndices)
                                z = sqrt(taup)*sum(conj(h(:,userIndices(k))).*v,1) + eta(1,userIndices(k));
                                if M_Appro==0
                                alphaEst_approx2 = exp(gammaln(M+1/2)-gammaln(M))^2*q*taup^2*rho*(betas(userIndices(k))./real(z)).^2-sigma2;
                                end
                                 if M_Appro==1
                                alphaEst_approx2 = M*q*taup^2*rho*(betas(userIndices(k))./real(z)).^2-sigma2;
                                 end  
%                                 alphaEst_approx2 = exp(gammaln(M+1/2)-gammaln(M))^2*q*taup^2*rho*(betas(userIndices(k))./real(z)).^2-sigma2;
                                if alphaEst_approx2<rho*betas(userIndices(k))*taup
                                    alphaEst_approx2 = rho*betas(userIndices(k))*taup;
                                end
                                if  RandomAccess==1
                                    if method == 1
                                        retransmit_3(k) =  rho*betas(userIndices(k))*taup>alphaEst_approx2/2-betas(userIndices(k))/sqrt(M);
                                    elseif method == 2
                                        retransmit_3(k) =  rho*betas(userIndices(k))*taup>(alphaEst_approx2-interCellBias)/2-betas(userIndices(k))/sqrt(M);
                                    end
                                end
                                if  RandomAccess==2
                                    randomNum = rand(1);
                                    if method == 1
                                        
                                        contendersEst_3(k) =  round(alphaEst_approx2/(rhoAverage*taup)); % Compute estimate of contenders at user k
                                        retransmit_3(k) = randomNum<(1/contendersEst_3(k));
                                    elseif method == 2
                                        
                                        contendersEst_3(k) =  round((alphaEst_approx2-interCellBias)/(rhoAverage*taup)); % Compute estimate of contenders at user k
                                        retransmit_3(k) = randomNum<(1/contendersEst_3(k));
                                    end
                                end
                                if or(RandomAccess==3,RandomAccess==4)
                                    %     ComValue = 4*rand(1);
                                    if method == 1
                                        %         SUCRe
                                        %         retransmit(k) =  rho*betas(userIndices(k))*taup...,
                                        %             >alphaEst_approx2/2-betas(userIndices(k))/sqrt(M);
                                        %         补偿因子CompRatio
                                        %         if UserDistance(userIndices(k)) < 1.1*cellRadius/2
                                        %
                                        %             if method == 1
                                        %                 retransmit(k) =  rho*betas(userIndices(k))*taup...,
                                        %                     >alphaEst_approx2/2-betas(userIndices(k))/sqrt(M);
                                        %             elseif method == 2
                                        %                 retransmit(k) =  rho*betas(userIndices(k))*taup...,
                                        %                     >(alphaEst_approx2-interCellBias)/2-betas(userIndices(k))/sqrt(M);
                                        %             end
                                        if betas(userIndices(k)) >0.99
                                            % retransmit(k) =  CompRatio*rho*betas(userIndices(k))*taup*10^(rhodB_Comp(userIndices(k))/10)...,
                                            retransmit_3(k) =  CompRatio*rho*betas(userIndices(k))*taup...,
                                                >alphaEst_approx2/2+biasDelta*betas(userIndices(k))/sqrt(M);
                                        else
                                            %  retransmit(k) =  rho*betas(userIndices(k))*taup*EdgeCompRatio*10^(rhodB_Comp(userIndices(k))/10)...,
                                            %  >alphaEst_approx2/2+biasDelta*betas(userIndices(k))/sqrt(M);
                                            retransmit_3(k) =  rho*betas(userIndices(k))*taup*10^(EdgeCompRatio*rhodB_Comp(userIndices(k))/10)...,
                                                >alphaEst_approx2/2+biasDelta*betas(userIndices(k))/sqrt(M);
                                        end
                                        %         原算法
                                        %         retransmit(k) =  rho*betas(userIndices(k))*taup*10^(CompRatio*rhodB_Comp(userIndices(k))/10)...,
                                        %             >alphaEst_approx2/2-betas(userIndices(k))/sqrt(M);
                                    elseif method == 2 % last fourth lines of Section IV.D
                                        %         SUCRe
                                        %         retransmit(k) =  rho*betas(userIndices(k))*taup...,
                                        %             >(alphaEst_approx2-Intra_Cell_Ratio*interCellBias)/2-betas(userIndices(k))/sqrt(M);
                                        %         补偿因子CompRatio
                                        %         if UserDistance(userIndices(k)) < 1.1*cellRadius/2
                                        if betas(userIndices(k)) >0.99
                                            %             retransmit(k) =  CompRatio*rho*betas(userIndices(k))*taup*10^(rhodB_Comp(userIndices(k))/10)...,
                                            retransmit_3(k) =  CompRatio*rho*betas(userIndices(k))*taup...,
                                                >(alphaEst_approx2-Intra_Cell_Ratio*interCellBias)/2+biasDelta*betas(userIndices(k))/sqrt(M);
                                        else
                                            retransmit_3(k) =  rho*betas(userIndices(k))*taup*10^(EdgeCompRatio*rhodB_Comp(userIndices(k))/10)...,
                                                >(alphaEst_approx2-Intra_Cell_Ratio*interCellBias)/2+biasDelta*betas(userIndices(k))/sqrt(M);
                                        end
                                        % 原算法
                                        % retransmit(k) =  rho*betas(userIndices(k))*taup*10^(CompRatio*rhodB_Comp(userIndices(k))/10)...,
                                        %   >(alphaEst_approx2-Intra_Cell_Ratio*interCellBias)/2-betas(userIndices(k))/sqrt(M);
                                        
                                    end
                                end
                                stop =1;
                            end
                            if sum(retransmit_3) == 1
                                successfulAccess(userIndices,3)=retransmit_3;
                            end
                        end
                    end
                end
                if L == 1
                    if method == 1 || method == 2
                        index_1 = find(successfulAccess(:,1)==1);
                        successfulAttempt(index_1) = true;
                        for i =1:length(index_1)%
                            finalWaitingTimes(accessAttempts(index_1(i)),indProb,method) = finalWaitingTimes(accessAttempts(index_1(i)),indProb,method) + 1;
                        end
                    end
                end
                if L == 2
                    if method == 1 || method == 2
                        index_1 = find(successfulAccess(:,1)==1);
                        index_2 = find(successfulAccess(:,2)==1);
                        index_1_2 = union(index_1,index_2);
                        successfulAttempt(index_1_2) = true;
                        for i =1:length(index_1_2)%
                            finalWaitingTimes(accessAttempts(index_1_2(i)),indProb,method) = finalWaitingTimes(accessAttempts(index_1_2(i)),indProb,method) + 1;
                        end
                    end
                end
                if L == 3
                    if method == 1 || method == 2
                        index_1 = find(successfulAccess(:,1)==1);
                        index_2 = find(successfulAccess(:,2)==1);
                        index_3 = find(successfulAccess(:,3)==1);
                        index_1_2_3 = union(union(index_1,index_2),union(index_1,index_3));
                        successfulAttempt(index_1_2_3) = true;
                        for i =1:length(index_1_2_3)%
                            finalWaitingTimes(accessAttempts(index_1_2_3(i)),indProb,method) = finalWaitingTimes(accessAttempts(index_1_2_3(i)),indProb,method) + 1;
                        end
                    end
                end
                stop =1;  
                %Determine which of the UEs that have failed too many times
                %with their access attempts and will give up
                giveUp = (accessAttempts(successfulAttempt==false) == maxAttempts);
                
                %Place the failing UEs at the last place in the vector of
                %waiting times for the failing UEs
                if L == 1 || L == 2 ||  L == 3
                if method == 1 || method ==2
                    finalWaitingTimes(end,indProb,method) = finalWaitingTimes(end,indProb,method) + sum(giveUp);
                elseif method == 3
                    finalWaitingTimesBaseline(end,indProb) = finalWaitingTimesBaseline(end,indProb) + sum(giveUp);
                end
                end
                
                %Keep the important parameters for all the UEs that failed
                %to access the network and has not given up
                waitingTime = accessAttempts((successfulAttempt==false) & (accessAttempts<maxAttempts));
                waitingBetas = betas((successfulAttempt==false) & (accessAttempts<maxAttempts));
                waitingAngles = UserAngles((successfulAttempt==false) & (accessAttempts<maxAttempts));
                waitingDistance = UserDistance((successfulAttempt==false) & (accessAttempts<maxAttempts));
                waitingIntercellInter = interCellVarianceDownlink((successfulAttempt==false) & (accessAttempts<maxAttempts));
                %   waitingUsers_Comp_index = Users_Comp_index((successfulAttempt==false) & (accessAttempts<maxAttempts));
                if  RandomAccess==3
                    waitingRhodB_Comp = rhodB_Comp((successfulAttempt==false) & (accessAttempts<maxAttempts));
                end
                
            end
            
            
            if mod(cellRadius-0.1*cellRadius,disInterval) == 0
                idxPower = ceil((newUserDistance-0.1*cellRadius)/disInterval);
                Disrange = 0.1*cellRadius+disInterval/2:disInterval:cellRadius-disInterval/2;
                for useridx = 1:newUsers(r)
                    powerCounts(idxPower(useridx)) = powerCounts(idxPower(useridx)) + 1;
                    if RandomAccess == 1
                        transPower(idxPower(useridx)) = transPower(idxPower(useridx)) + rhodB(1);
                    else
                        transPower(idxPower(useridx)) = transPower(idxPower(useridx)) + rhodB(useridx);
                    end
                end
            else
%                 warning(['Distance interval is set uncorrectly.']);
                powerCounts = powerCounts;
            end
            
        end
        meanWaitingTime(indProb,method) = (([1:maxAttempts maxAttempts])*finalWaitingTimes(:,indProb,method))/sum(finalWaitingTimes(:,indProb,method));
    end
end
% %Compute the average number of access attempts that the UEs make
% meanWaitingTime = zeros(length(K0values),2);
% % meanWaitingTimeBaseline = zeros(length(K0values),1);
% for indProb = 1:length(K0values)
%     meanWaitingTime(indProb,1) = (([1:maxAttempts maxAttempts])*finalWaitingTimes(:,indProb,1))/(sum(finalWaitingTimes(:,indProb,1))+eps);
%     meanWaitingTime(indProb,2) = (([1:maxAttempts maxAttempts])*finalWaitingTimes(:,indProb,2))/(sum(finalWaitingTimes(:,indProb,2))+eps);
% %     meanWaitingTimeBaseline(indProb) = (([1:maxAttempts maxAttempts])*finalWaitingTimesBaseline(:,indProb))/(sum(finalWaitingTimesBaseline(:,indProb))+eps);
% end
% for indProb = 1:length(K0values)
%     meanCollidingUsersNum(indProb,1) = mean(NumberOfCollidingUsers(:,indProb,1))/taup;
%     meanCollidingUsersNum(indProb,2) = mean(NumberOfCollidingUsers(:,indProb,2))/taup;
% end
DataAll = meanWaitingTime.';
% DataAll(end+1,:) = meanWaitingTimeBaseline.';
DisPower = (transPower ./ powerCounts) / rhoSUCRedB;
end
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