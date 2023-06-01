%Initialization
close all; clear all;
SUCRe =1;   SUCRe_range =[ 0 0];
RandomSeed = 0; Kneighboringcells = 10; nbrOfRAblocks = 113; SumIndex =0;
for SUCRe =SUCRe_range
    if RandomSeed ==0 rand('state',12345);  randn('state',12345*3);  end% 保证每次SNR循环，初始种子都一样
    
    [K0values,finalWaitingTimesACBPC,meanWaitingTimeACBPC,meanWaitingTimeBaseline,finalWaitingTimesBaseline,meanCollidingUsersNumACBPC] = ACPC_func(SUCRe,Kneighboringcells,nbrOfRAblocks)
    %   finalWaitingTimesACBPC_all{i} = finalWaitingTimesACBPC;
    SumIndex = SumIndex+1;
    if SumIndex ==1
        DataMatrix =meanWaitingTimeACBPC.';
    else
        DataMatrix(size(DataMatrix,1)+1: size(DataMatrix,1)+size(meanWaitingTimeACBPC,2),:) =meanWaitingTimeACBPC';
    end
end
MarkerSize=9; LineWidth =2; LineMethod =1; PlotMethod =0; FontSize=22;
DataMatrix(end+1,:) =meanWaitingTimeBaseline.';
SNR_Matrix =repmat(K0values,size(DataMatrix,1),1);
NumofFiles =size(DataMatrix,1);

Plot_SIC_results=Plot_SIC(MarkerSize,LineWidth,FontSize,SNR_Matrix,DataMatrix,NumofFiles,LineMethod,PlotMethod)'
figure;

DataMatrix
plot(K0values,meanWaitingTimeACBPC(:,1),'r-+','LineWidth',1);
plot(K0values,meanWaitingTimeACBPC(:,2),'b-+','LineWidth',1);
plot(K0values,meanWaitingTimeBaseline,'k--','LineWidth',1);
xlabel('Number of Inactive UEs');
ylabel('Average Number of Access Attempts');
legend('ACBPC: No interf.','ACBPC: Interf.','Baseline','Location','SouthEast');
ylim([0 10]);

PlotData1
  
%   
%Set the number of Monte-Carlo realizations
% nbrOfRAblocks = 50000;
nbrOfRAblocks = 4150;
%Number of BS antennas
M = 100;
%Range of number of inactive UEs in the cell
K0values = [100 250 500 1000:6000:28000];
% K0values = 5000;
factor = 0.9; % 功率降低指数
%Probability that an inactive UE wants to become active in a given block
pA = 0.1;
%Maximum number of attempts to send RA pilots before a UE gives up
maxAttempts = 10;
%Probability of retransmitting an RA pilot in each follow RA block, when
%the first transmission was failed.
tryAgainProb = 0.5;
%Number of RA pilot signals
taup = 10;
%%Define simulation scenario
q = 1; %Transmit power of the BS
sigma2 = 1; %Noise variance
rhoSUCRe = 1; %Transmit power of UEs in the cell
rho = 1;
rhoAverage = sigma2;
%Standard deviation of shadow fading
%shadowFadingStddB = 10;
shadowFadingStddB = 8;
%Set cell radius (in meter)
cellRadius = 250;
rhoSUCRedB = - 98.65 + 35.3 + 38*log10(cellRadius);
%Generate noise realizations at the BS in the center cell 中心小区基站的噪声
n = sqrt(1/2)*(randn(M,taup,nbrOfRAblocks)+1i*randn(M,taup,nbrOfRAblocks));
%Matrices to store the simulation results, in terms of waiting time for the
%SUCRe protocol (without or with inter-cell interference) and baseline scheme
finalWaitingTimes = zeros(maxAttempts+1,length(K0values),2);
finalWaitingTimesBaseline = zeros(maxAttempts+1,length(K0values));
NumberOfCollidingUsers = zeros(nbrOfRAblocks,length(K0values),2) ;
%Set number of active UEs in the neighboring cells 设置相邻小区中活跃用户的数量
Kneighboringcells = 10;
%Set transmit power of users in adjacent cells 设置相邻小区用户的传输功率
rhoIntercell = 1;
%Compute locations of all neighboring BSs 计算所有邻近BSs的位置
neighboringBSs = 2*cellRadius*sqrt(3)/2 * exp(1i*(0:5)*pi/3);
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
            notReady(j,:) = (shadowFadingStddB*shadowFadingRealizationsWithinOwnCellUplink(k,:,j) - 38*log10(abs(userLocationsNeighboring(k,:,j))) < shadowFadingStddB*shadowFadingRealizationsIntercellUplink(k,:,j) - 38*log10(abs(userLocationsNeighboring(k,:,j)-neighboringBSs(j))) ); 
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
    POWER = -( 98.65 - 35.3 - 38*log10(abs(userLocationsNeighboring(:,:,j) + neighboringBSs(j))) + shadowFadingStddB*shadowFadingRealizationsIntercellUplink(:,:,j)  );
    lower_bound = 20;
    index = find(POWER>lower_bound); 
    POWER(index) = POWER(index).^factor;
    interCellVarianceUplink = interCellVarianceUplink + sum(rhoIntercell*10.^( ( POWER+ 98.65 - 35.3 - 38*log10(abs(userLocationsNeighboring(:,:,j) + neighboringBSs(j))) + shadowFadingStddB*shadowFadingRealizationsIntercellUplink(:,:,j)  )/10   ),1) ;    
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
    stop = 1;    
    %There are three methods that are considered:
    %1: SUCRe without inter-cell interference
    %2: SUCRe with inter-cell interference
    %3: Baseline scheme
    for method = 1:3
        %Initiate the set of UEs that have failed to access the network
        waitingTime = []; %Contains number of access attempts
        waitingBetas = []; %Channel variance of the users
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
            newShadowFading = randn(newUsers(r),1);   %Generate shadow fading realizations
            shadowFadingRealizationsIntercellDownlink = randn(newUsers(r),length(neighboringBSs));   %Generate shadow fading realizations for downlink inter-cell interference
            %Go through all new users and make sure that they are always
            %served by the BS in the own hexagonal cell (even when shadow
            %fading is taken into account)
            notReady = 1;
            while sum(notReady)>0
                notReady = zeros(newUsers(r),1);
                for j = 1:length(neighboringBSs)
                    %Check which of the users that are served by the right BS
                    notReady = notReady + (shadowFadingStddB*newShadowFading - 38*log10(newUserDistance) < shadowFadingStddB*shadowFadingRealizationsIntercellDownlink(:,j) - 38*log10(abs(newUserLocations-neighboringBSs(j))) );
                end
                %Make new random shadow fading realizations for those users that
                %have a better channel to the neighboring BS
                newShadowFading(notReady>0) = randn(sum(notReady>0),1);
                shadowFadingRealizationsIntercellDownlink(notReady>0,:) = randn(sum(notReady>0),length(neighboringBSs));
            end
            
            %Compute average signal gain for non-line-of-sight propagation
            %( -98.65 dBm represents the noise variance)
            % rhoSUCRedB = - 98.65 + 35.3 + 38*log10(cellRadius) % THE FOURTH
            rhoNewdB = - 98.65 + 35.3 + 38*log10(newUserDistance)- shadowFadingStddB*newShadowFading;
            lower_bound = 20;
            rhoNewdB_R = rhoNewdB;
%             index = zeros();
            index = find(rhoNewdB_R>lower_bound); % &rhoNewdB>0)
            newUsers_Comp_index = index;
            
            rhoNewdB_R(index) = rhoNewdB_R(index).^factor;
            newRhodB_Comp = rhoNewdB - rhoNewdB_R; % 记录降低的功率值（绝对值）
            rhodB = rhoNewdB_R;
            newBetas = 10.^( (rhodB+ 98.65 - 35.3 - 38*log10(newUserDistance) + shadowFadingStddB*newShadowFading  )/10   );         
            %Compute the total inter-cell interference in the uplink and downlink in
            %each channel realization
            newIntercellVarianceDownlink = zeros(newUsers(r),1);
            for j = 1:length(neighboringBSs)
                %Note: -98.65 dBm represents the noise variance
                POWER = -( 98.65 - 35.3 - 38*log10(abs(newUserLocations-neighboringBSs(j))) + shadowFadingStddB*shadowFadingRealizationsIntercellDownlink(:,j)  ); 
                lower_bound = 20;
                index = find(POWER>lower_bound);
                POWER(index) = POWER(index).^factor;
                newIntercellVarianceDownlink = newIntercellVarianceDownlink + qNeighbor*10.^( ( POWER + 98.65 - 35.3 - 38*log10(abs(newUserLocations-neighboringBSs(j))) + shadowFadingStddB*shadowFadingRealizationsIntercellDownlink(:,j)  )/10   );
            end
            interCellVarianceDownlink = [newIntercellVarianceDownlink; waitingIntercellInter];
            Users_Comp_index = [newUsers_Comp_index ;waitingUsers_Comp_index];
            rhodB_Comp = [newRhodB_Comp; waitingRhodB_Comp]; % 组合新用户和未成功接入用户降低的功率值        
            %Combine the new UEs with the ones that have made previous access attempts
            betas = [newBetas; waitingBetas];
            numberOfAccessingUsers = length(betas);  %Compute number of UEs that will send pilots
            
            %Randomize if each of the UEs that retransmit pilots should
            %really send a pilot in this RA block. One means retransmit and
            %zero means do not retransmit in this block
            shouldWaitingUsersRetransmit = binornd(1,tryAgainProb,size(waitingTime));
            
            %Create a list of the UEs that will send pilots (all new UEs
            %transmit pilots)
            accessAttempt = [ones(newUsers(r),1) ; shouldWaitingUsersRetransmit]; %新用户+失败用户
            
            %Randomize which of the pilots that each of the UEs are using
            pilotSelections = accessAttempt.*randi(taup,[numberOfAccessingUsers 1]);
            
            %Count the number of pilots that each of the UEs will have
            %transmitted, after this block
            accessAttempts = [ones(newUsers(r),1); waitingTime+shouldWaitingUsersRetransmit];  %重传的次数
            stop =1 ;
            
            %Check if there is at least on UE that transmit pilots in this RA block
            if ~isempty(accessAttempts)
                
                %Generate uncorrelated Rayleigh fading channel realizations
                h = (randn(M,numberOfAccessingUsers)+1i*randn(M,numberOfAccessingUsers));
                h = repmat(sqrt(betas'/2),[M 1]) .* h;
                % rho = min(rhoAverage./betas,rhoSUCRe);
                
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
                
                
                %Go through all RA pilots
                for t = 1:taup
                    
                    %Extract the UE that transmit pilot t
                    userIndices = find(pilotSelections==t);
                    
                    if ~isempty(userIndices)
                        %Consider the SUCRe protocol
                        if method == 1 || method == 2
                            if length(userIndices)>1
                                NumberOfCollidingUsers(r,indProb,method) = NumberOfCollidingUsers(r,indProb,method)+length(userIndices);
                            end
                            %Compute the received signal in Eq. (6)
                            % yt = sqrt(taup)* sum( repmat(sqrt(rho(userIndices))',[M 1]) .* h(:,userIndices),2) + sqrt(noiseInterfvariance)*n(:,t,r);
                            yt = sqrt(taup*rho) * sum(h(:,userIndices),2) + sqrt(noiseInterfvariance)*n(:,t,r);
                            
                            %Compute the precoding vector used by the BS
                            v = sqrt(q)*yt/norm(yt);
                            
                            
                            %Prepare a list of UEs that decide to retransmit
                            %pilot t
                            retransmit = false(length(userIndices),1);
                            contendersEst = zeros(length(userIndices),1); 
                            %Go through the UEs that transmitted pilot t
                            for k = 1:length(userIndices)   
                                %Compute the received DL signal at user k in Eq. (13)
                                z = sqrt(taup)*sum(conj(h(:,userIndices(k))).*v,1) + eta(1,userIndices(k)); 
                                %Compute estimate of alpha_t at user k using Approx2 in Eq. (36)
                                alphaEst_approx2 = exp(gammaln(M+1/2)-gammaln(M))^2*q*taup^2*rho*(betas(userIndices(k))./real(z)).^2-sigma2;
                                
                                if alphaEst_approx2<rho*betas(userIndices(k))*taup
                                    alphaEst_approx2 = rho*betas(userIndices(k))*taup;
                                end
                                
                                %Apply the retransmission decision rule with a
                                %bias term of minus one standard deviation and
                                %the average uplink inter-cell interference                               
                                if method == 1
%                                     if ~isempty(find(Users_Comp_index==userIndices(k)))
%                                     retransmit(k) =  rho*betas(userIndices(k))*taup...,
%                                         *10^(rhodB_Comp(Users_Comp_index(find(Users_Comp_index==userIndices(k))))/10)...,
%                                         >alphaEst_approx2/2-betas(userIndices(k))/sqrt(M);
%                                     else
%                                         retransmit(k) =  rho*betas(userIndices(k))*taup...,
%                                             >alphaEst_approx2/2-betas(userIndices(k))/sqrt(M);
%                                     end
                                    retransmit(k) =  rho*betas(userIndices(k))*taup...,
                                        *10^(rhodB_Comp(userIndices(k))/10)...,
                                        >alphaEst_approx2/2-betas(userIndices(k))/sqrt(M);
                                elseif method == 2
                                     retransmit(k) =  rho*betas(userIndices(k))*taup...,
                                        *10^(rhodB_Comp(userIndices(k))/10)...,
                                        >(alphaEst_approx2-interCellBias)/2-betas(userIndices(k))/sqrt(M);                                  
%                                     if ~isempty(find(Users_Comp_index==userIndices(k)))
%                                     retransmit(k) =  rho*betas(userIndices(k))*taup*10^rhodB_Comp(Users_Comp_index(find(Users_Comp_index==userIndices(k))))...,
%                                         >(alphaEst_approx2-interCellBias)/2-betas(userIndices(k))/sqrt(M);
%                                     else
%                                     retransmit(k) =  rho*betas(userIndices(k))*taup...,
%                                         >(alphaEst_approx2-interCellBias)/2-betas(userIndices(k))/sqrt(M);
%                                     end
                                end
                            end
                            
                            %Check if only one UE has decided to retransmit
                            %pilot t and then admit the UE for data
                            %transmission and store the number of access
                            %attempts that the UE made.
                            if sum(retransmit) == 1
                                
                                successfulAttempt(userIndices(retransmit)) = true;
                                finalWaitingTimes(accessAttempts(retransmit),indProb,method) = finalWaitingTimes(accessAttempts(retransmit),indProb,method) + 1;
                                
                            end
                            
                            
                            %Consider the baseline scheme
                        elseif method == 3
                            
                            %Check if only one UE has transmitted pilot t and
                            %then admit the UE for data transmission and store
                            %the number of access attempts that the UE made.
                            if length(userIndices) == 1
                                
                                successfulAttempt(userIndices) = true;
                                finalWaitingTimesBaseline(accessAttempts(userIndices),indProb) = finalWaitingTimesBaseline(accessAttempts(userIndices),indProb) + 1;
                                
                            end
                        end
                    end
                end
                
                %Determine which of the UEs that have failed too many times
                %with their access attempts and will give up
                giveUp = (accessAttempts(successfulAttempt==false) == maxAttempts);
                
                %Place the failing UEs at the last place in the vector of
                %waiting times for the failing UEs
                if method == 1 || method ==2   
                    finalWaitingTimes(end,indProb,method) = finalWaitingTimes(end,indProb,method) + sum(giveUp); 
                elseif method == 3
                    
                    finalWaitingTimesBaseline(end,indProb) = finalWaitingTimesBaseline(end,indProb) + sum(giveUp);
                    
                end
                
                %Keep the important parameters for all the UEs that failed
                %to access the network and has not given up
                waitingTime = accessAttempts((successfulAttempt==false) & (accessAttempts<maxAttempts));
                waitingBetas = betas((successfulAttempt==false) & (accessAttempts<maxAttempts));
                waitingIntercellInter = interCellVarianceDownlink((successfulAttempt==false) & (accessAttempts<maxAttempts));
%                 waitingUsers_Comp_index = Users_Comp_index((successfulAttempt==false) & (accessAttempts<maxAttempts));
                waitingRhodB_Comp = rhodB_Comp((successfulAttempt==false) & (accessAttempts<maxAttempts));
                
            end
        end 
    end   
end
%Compute the average number of access attempts that the UEs make
meanWaitingTime = zeros(length(K0values),2);
meanWaitingTimeBaseline = zeros(length(K0values),1);

for indProb = 1:length(K0values)
    
    meanWaitingTime(indProb,1) = (([1:maxAttempts maxAttempts])*finalWaitingTimes(:,indProb,1))/sum(finalWaitingTimes(:,indProb,1));
    meanWaitingTime(indProb,2) = (([1:maxAttempts maxAttempts])*finalWaitingTimes(:,indProb,2))/sum(finalWaitingTimes(:,indProb,2));
    meanWaitingTimeBaseline(indProb) = (([1:maxAttempts maxAttempts])*finalWaitingTimesBaseline(:,indProb))/sum(finalWaitingTimesBaseline(:,indProb));
    
end
for indProb = 1:length(K0values)
   meanCollidingUsersNum(indProb,1) = mean(NumberOfCollidingUsers(:,indProb,1))/taup;
   meanCollidingUsersNum(indProb,2) = mean(NumberOfCollidingUsers(:,indProb,2))/taup;

end


%%Plot simulation results
figure
%%
subplot(121)
hold on; box on;
plot(K0values,meanWaitingTime(:,1),'r-.+','LineWidth',1);
plot(K0values,meanWaitingTime(:,2),'b-.+','LineWidth',1);
plot(K0values,meanWaitingTimeBaseline,'k--','LineWidth',1);
xlabel('Number of Inactive UEs');
ylabel('Average Number of Access Attempts');
legend('ACBPC: No interf.','ACBPC: Interf.','Baseline','Location','SouthEast');
ylim([0 10]);
 
subplot(122)
hold on; box on;
yyaxis left;
plot(K0values,finalWaitingTimes(end,:,1)./sum(finalWaitingTimes(:,:,1),1),'r-.+','LineWidth',1);
plot(K0values,finalWaitingTimes(end,:,2)./sum(finalWaitingTimes(:,:,2),1),'b-.+','LineWidth',1);
plot(K0values,finalWaitingTimesBaseline(end,:)./sum(finalWaitingTimesBaseline,1),'k--','LineWidth',1);
xlabel('Number of Inactive UEs');
ylabel('Fraction of Failed Access Attempts');
%legend('ACBPC: No interf.','ACBPC: Interf.','Baseline','Location','SouthEast');

yyaxis right;
plot(K0values,meanCollidingUsersNum(:,1),'r--x','LineWidth',1);
plot(K0values,meanCollidingUsersNum(:,2),'b--x','LineWidth',1);
xlabel('Number of Inactive UEs');
ylabel('Average Number of Colliding UEs');
% legend('ACBPC: No interf.','ACBPC: Interf.','Baseline','ACBPC: No interf.','ACBPC: Interf.','Location','SouthEast');
ylim([0 40])
