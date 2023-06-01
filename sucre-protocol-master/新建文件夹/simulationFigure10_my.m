% %This Matlab script can be used to generate Figure 10, in the article:
% Emil Bjornson, Elisabeth de Carvalho, Jesper H. Sorensen, Erik G. Larsson,
%Petar Popovski, "A Random Access Protocol for Pilot Allocation in Crowded
% %Massive MIMO Systems," IEEE Transactions on Wireless Communications, To appear.
%  Download article: http://arxiv.org/pdf/1604.04248
% This is version 1.1 (Last edited: 2019-01-31)
%%License: This code is licensed under the GPLv2 license. If you in any way
%use this code for research that results in publications, please cite our
%original article listed above.
%Initialization
close all; clear all;
%Set the number of Monte-Carlo realizations
nbrOfRAblocks = 2500; % 50000£»
%Number of BS antennas
M = 6;
%Range of number of inactive UEs in the cell
K0values = [100 250 500:250:12000];
K0values = [100 250 500:2500:12000];

% K0values = [10000];

%Probability that an inactive UE wants to become active in a given block
pA = 0.001;
%Maximum number of attempts to send RA pilots before a UE gives up
maxAttempts = 10;
%Probability of retransmitting an RA pilot in each follow RA block, when
%the first transmission was failed.
tryAgainProb = 0.5;
%Number of RA pilot signals
taup = 10;
%%Define simulation scenario
rho = 1; %Transmit power of UEs in the cell
q = 1; %Transmit power of the BS
sigma2 = 1; %Noise variance
%Standard deviation of shadow fading
shadowFadingStddB = 10;
%Set cell radius (in meter)
cellRadius = 250;
%Generate noise realizations at the BS in the center cell
n = sqrt(1/2)*(randn(M,taup,nbrOfRAblocks)+1i*randn(M,taup,nbrOfRAblocks));
%Matrices to store the simulation results, in terms of waiting time for the
%SUCRe protocol (without or with inter-cell interference) and baseline scheme
finalWaitingTimesSUCRe = zeros(maxAttempts+1,length(K0values),2);
finalWaitingTimesBaseline = zeros(maxAttempts+1,length(K0values));

%Set number of active UEs in the neighboring cells
Kneighboringcells = 10;
%Set transmit power of users in adjacent cells
rhoIntercell = 1;
%Compute locations of all neighboring BSs
neighboringBSs = 2*cellRadius*sqrt(3)/2 * exp(1i*(0:5)*pi/3);

%Generate user locations in neighboring cells
userLocationsNeighboring = generatePointsHexagon([Kneighboringcells,nbrOfRAblocks,length(neighboringBSs)],cellRadius,0.1*cellRadius);
% figure; 
% %  hold on; 
% plt.figure(figsize = (16, 12))
% ax = plt.subplot(1,2,1)
% sns.relplot(x='total_bill', y='tip', data=tips, ax= ax);
% ax = plt.subplot(1,2,2)
% scatterplot(userLocationsNeighboring(1,:,:)); 
%  figure; scatterplot(userLocationsNeighboring(2,:,:)); 
%  figure; subplot(2,2,1); scatterplot(userLocationsNeighboring(6,:,:)); 
%Generate shadow fading realizations of users in neighboring cells, both
%within that cell and to the cell under study
shadowFadingRealizationsWithinOwnCellUplink = randn([Kneighboringcells,nbrOfRAblocks,length(neighboringBSs)]);
shadowFadingRealizationsIntercellUplink = randn([Kneighboringcells,nbrOfRAblocks,length(neighboringBSs)]);
%Go through all users in neighboring cells
for k = 1:Kneighboringcells    
    notReady = 1;    
    while sum(notReady)>0        
        notReady = zeros(length(neighboringBSs),nbrOfRAblocks);        
        for j = 1:length(neighboringBSs)            
            %Check which of the users that are served by the right BS %scatterplot(neighboringBSs); 
            notReady(j,:) = (shadowFadingStddB*shadowFadingRealizationsWithinOwnCellUplink(k,:,j) ...,
                - 38*log10(abs(userLocationsNeighboring(k,:,j)))...,
                < shadowFadingStddB*shadowFadingRealizationsIntercellUplink(k,:,j)...,
                - 38*log10(abs(userLocationsNeighboring(k,:,j)-neighboringBSs(j))) );            
            %Make new random shadow fading realizations for those users in
            %the neighborin cell that have a better channel to the center BS
            shadowFadingRealizationsWithinOwnCellUplink(k,notReady(j,:)>0,j) = randn(1,sum(notReady(j,:)>0));
            shadowFadingRealizationsIntercellUplink(k,notReady(j,:)>0,j) = randn(1,sum(notReady(j,:)>0));            
        end        
    end    
end
%Compute the total inter-cell interference in the uplink and downlink in
%each channel realization
interCellVarianceUplink = zeros(1,nbrOfRAblocks);
for j = 1:length(neighboringBSs)    
    %Note: 27 dBm represents the transmit power and -98.65 dBm represents
    %the noise variance
    interCellVarianceUplink = interCellVarianceUplink + sum(rhoIntercell*10.^( (27+ 98.65 - 34.53 - 38*log10(abs(userLocationsNeighboring(:,:,j) + neighboringBSs(j))) + shadowFadingStddB*shadowFadingRealizationsIntercellUplink(:,:,j))/10 ),1) ;    
end
%Compute the average uplink inter-cell interference, which is used in the
%bias terms
interCellBias = mean(interCellVarianceUplink);
%Go through all different number of inactive UEs
for indProb = 1:length(K0values)    
    %Display simulation progress
    disp(['K0 values: ' num2str(indProb) ' out of ' num2str(length(K0values))]);    
    %Extract current value of the number of inactive UEs
    K0 = K0values(indProb);    
    %Generate the number of UEs that wish to access the network (for the first time) in each of the RA blocks
    newUsers = binornd(K0,pA,[nbrOfRAblocks 1]);    
    %There are three methods that are considered:
    %1: SUCRe without inter-cell interference; 2: SUCRe with inter-cell interference; 3: Baseline scheme
    for method =1:3        
        %Initiate the set of UEs that have failed to access the network
        waitingTime = []; %Contains number of access attempts
        waitingBetas = []; %Channel variance of the users
        waitingIntercellInter = []; %Downlink inter-cell inteference variance        
        %Set the inter-cell interference parameters
        if method == 2            
            qNeighbor = taup*q; %Transmit power of neighbouring BSs
            rhoIntercell = rho; %Set transmit power of users in adjacent cells            
        else            
            qNeighbor = 0; %No transmit power of neighbouring BSs
            rhoIntercell = 0; %No transmit power of users in adjacent cells            
        end        
        %Go through all RA blocks that are considered in the Monte-Carlo simulations
        for r = 1:nbrOfRAblocks            
            %Generate locations of new UEs that try to access the network
            newUserLocations = generatePointsHexagon([newUsers(r) 1],cellRadius,0.1*cellRadius);
            %Extract user distances from the BS scatterplot(newUserLocations);
            newUserDistance = abs(newUserLocations);  
            %Generate shadow fading realizations
            newShadowFading = randn(newUsers(r),1);   
            %Generate shadow fading realizations for downlink inter-cell interference
            shadowFadingRealizationsIntercellDownlink = randn(newUsers(r),length(neighboringBSs));            
            %Go through all new users and make sure that they are always served by the BS in the own
            %hexagonal cell (even when shadow fading is taken into account)            
            notReady = 1;            
            while sum(notReady)>0                
                notReady = zeros(newUsers(r),1);                
                for j = 1:length(neighboringBSs)                    
                    %Check which of the users that are served by the right BS
                    notReady = notReady + (shadowFadingStddB*newShadowFading - 38*log10(newUserDistance) < shadowFadingStddB*shadowFadingRealizationsIntercellDownlink(:,j) - 38*log10(abs(newUserLocations-neighboringBSs(j))) );
                end
                %Make new random shadow fading realizations for those users that have a better channel to the neighboring BS
                newShadowFading(notReady>0) = randn(sum(notReady>0),1);
                shadowFadingRealizationsIntercellDownlink(notReady>0,:) = randn(sum(notReady>0),length(neighboringBSs));                
            end
            %Compute average signal gain for non-line-of-sight propagation (27 dBm represents the transmit power
            %and -98.65 dBm represents the noise variance)
            newBetas = 10.^((27+ 98.65 - 34.53 - 38*log10(newUserDistance) + shadowFadingStddB*newShadowFading)/10 );   
            EstBeta = (98.65 - 34.53 - 38*log10(newUserDistance) + shadowFadingStddB*newShadowFading); 
            trans_power = 27 ;
            EstBeta1 = 10.^((98.65 - 34.53 - 38*log10(newUserDistance) + shadowFadingStddB*newShadowFading)/10 );             
%           newBetas1 = 10.^((17+ 98.65 - 34.53 - 38*log10(newUserDistance) + shadowFadingStddB*newShadowFading)/10 );     
%           plot(newBetas1); hold on;  plot(newBetas,'r'); 
            %Compute the total inter-cell interference in the uplink and downlink in each channel realization
            newIntercellVarianceDownlink = zeros(newUsers(r),1);
            newIntercellVarianceDownlink1 = zeros(newUsers(r),1);      
            for j = 1:length(neighboringBSs)                
                %Note: 27 dBm represents the transmit power and -98.65 dBm represents the noise variance
                newIntercellVarianceDownlink = newIntercellVarianceDownlink + qNeighbor*10.^( (27+ 98.65 - 34.53 - 38*log10(abs(newUserLocations-neighboringBSs(j))) + shadowFadingStddB*shadowFadingRealizationsIntercellDownlink(:,j))/10 );                
                % newIntercellVarianceDownlink1 = newIntercellVarianceDownlink1 + qNeighbor*10.^( (17+ 98.65 - 34.53 - 38*log10(abs(newUserLocations-neighboringBSs(j))) + shadowFadingStddB*shadowFadingRealizationsIntercellDownlink(:,j))/10 );                
            end            
            interCellVarianceDownlink = [newIntercellVarianceDownlink; waitingIntercellInter];    
            %Combine the new UEs with the ones that have made previous access attempts
            betas = [newBetas; waitingBetas];    
            if  size(waitingBetas,1)>0
                waitingBetas;
            end
            
            %Compute number of UEs that will send pilots
            numberOfAccessingUsers = length(betas);            
            %Randomize if each of the UEs that retransmit pilots should
            %really send a pilot in this RA block. One means retransmit and
            %zero means do not retransmit in this block
            shouldWaitingUsersRetransmit = binornd(1,tryAgainProb,size(waitingTime)); 
            
            %Create a list of the UEs that will send pilots (all new UEs transmit pilots)
            accessAttempt = [ones(newUsers(r),1) ; shouldWaitingUsersRetransmit];            
            %Randomize which of the pilots that each of the UEs are using
            pilotSelections = accessAttempt.*randi(taup,[numberOfAccessingUsers 1]);            
            %Count the number of pilots that each of the UEs will have
            %transmitted, after this block
            accessAttempts = [ones(newUsers(r),1); waitingTime+shouldWaitingUsersRetransmit];            
            %Check if there is at least on UE that transmit pilots in this RA block
            if ~isempty(accessAttempts)                
                %Generate uncorrelated Rayleigh fading channel realizations
                h = (randn(M,numberOfAccessingUsers)+1i*randn(M,numberOfAccessingUsers));
                h = repmat(sqrt(betas'/2),[M 1]) .* h;                
                %Generate noise plus inter-cell interference realizations at the UEs.
                if method == 1
                    noiseInterfvariance = sigma2;
                    eta = sqrt(sigma2/2)*(randn(1,numberOfAccessingUsers)+1i*randn(1,numberOfAccessingUsers));
                elseif method == 2 
                    noiseInterfvariance = sigma2+interCellVarianceUplink(1,r);% last two terms of (13)
                    eta = sqrt((sigma2+interCellVarianceDownlink')/2).*(randn(1,numberOfAccessingUsers)+1i*randn(1,numberOfAccessingUsers));
                    % effective noise = noise +interference  extension of  last two terms of (13)
                end           
                %Prepare a list of UEs that succeed in the random access
                successfulAttempt = false(size(betas));  % ?             
                %Go through all RA pilots
                for t = 1:taup                    
                    %Extract the UE that transmit pilot t
                    userIndices = find(pilotSelections==t); 
                    %Consider the SUCRe protocol
                    if method == 1 || method == 2                        
                        %Compute the received signal in Eq. (6)
                        yt = sqrt(taup*rho) * sum(h(:,userIndices),2) + sqrt(noiseInterfvariance)*n(:,t,r);
                        %Compute the precoding vector used by the BS
                        v = sqrt(q)*yt/norm(yt);    %sqrt(sum(abs(yt).^2))                     
                        %Prepare a list of UEs that decide to retransmit pilot t
                        retransmit = false(length(userIndices),1);                        
                        %Go through the UEs that transmitted pilot t
                        for k = 1:length(userIndices)                            
                            %Compute the received DL signal at user k in Eq. (13)
                            z = sqrt(taup)*sum(conj(h(:,userIndices(k))).*v,1) + eta(1,userIndices(k));                            
                            %Compute estimate of alpha_t at user k using Approx2 in Eq. (36)
                            alphaEst_approx2 = exp(gammaln(M+1/2)-gammaln(M))^2*q*taup^2*rho*(betas(userIndices(k))./real(z)).^2-sigma2;
                            %Compute estimate of alpha_t at user k using Approx2 in Eq. (36)
                            if alphaEst_approx2<rho*betas(userIndices(k))*taup
                                alphaEst_approx2 = rho*betas(userIndices(k))*taup;
                            end                            
                            %Apply the retransmission decision rule with a
                            %bias term of minus one standard deviation and
                            %the average uplink inter-cell interference
                            biasDelta =[0];
                              for b = 1:length(biasDelta)    
                            if method == 1  % Eq. (18) and the   fifth line from the bottom  Section IV.D.
                                retransmit(k) =  rho*betas(userIndices(k))*taup>alphaEst_approx2/2-betas(userIndices(k))/sqrt(M); %+ f(betas(userIndices(k)));
                                retransmit_bias(b,k) =  rho*betas(userIndices(k))*taup>alphaEst_approx2/2-rho*betas(userIndices(k))/sqrt(M); %+ f(betas(userIndices(k)));
                            elseif method == 2 % Eq. (18) and the fifth line from the bottom  Section IV.D.
                                retransmit(k) =  rho*betas(userIndices(k))*taup>(alphaEst_approx2-interCellBias)/2-betas(userIndices(k))/sqrt(M) ;
                                retransmit_bias(b,k) =  rho*betas(userIndices(k))*taup>(alphaEst_approx2-interCellBias)/2-betas(userIndices(k))/sqrt(M) ;%+func(betas(userIndices(k)));
                                
%                          retransmit(k,:,b) = rho(1,:,k).*betas(1,:,k)*taup>(alphaEst_approx2-interCellBias)/2+...,
%                         rho(1,:,k).*betas(1,:,k)*taup*biasDelta(b)/sqrt(Mvalues(mInd));   
%                                 
                            end                            
                              end
                        end                        
                        %Check if only one UE has decided to retransmit pilot t and then admit the UE for data
                        %transmission and store the number of access attempts that the UE made.
                        if sum(retransmit) == 1                            
                            successfulAttempt(userIndices(retransmit)) = true;
                            finalWaitingTimesSUCRe(accessAttempts(retransmit),indProb,method) = finalWaitingTimesSUCRe(accessAttempts(retransmit),indProb,method) + 1;                            
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
                %Determine which of the UEs that have failed too many times
                %with their access attempts and will give up
                giveUp = (accessAttempts(successfulAttempt==false) == maxAttempts);                
                %Place the failing UEs at the last place in the vector of
                %waiting times for the failing UEs
                if method == 1 || method ==2                    
                    finalWaitingTimesSUCRe(end,indProb,method) = finalWaitingTimesSUCRe(end,indProb,method) + sum(giveUp);                    
                elseif method == 3                    
                    finalWaitingTimesBaseline(end,indProb) = finalWaitingTimesBaseline(end,indProb) + sum(giveUp);                    
                end                
                %Keep the important parameters for all the UEs that failed
                %to access the network and has not given up
                waitingTime = accessAttempts((successfulAttempt==false) & (accessAttempts<maxAttempts));
                waitingBetas = betas((successfulAttempt==false) & (accessAttempts<maxAttempts));
                waitingIntercellInter = interCellVarianceDownlink((successfulAttempt==false) & (accessAttempts<maxAttempts));
            end            
        end       
    end    
end
%Compute the average number of access attempts that the UEs make
meanWaitingTimeSUCRe = zeros(length(K0values),2);
meanWaitingTimeBaseline = zeros(length(K0values),1);
for indProb = 1:length(K0values)
    finalWaitingTimesSUCRemeanWaitingTimeSUCRe(indProb,1) = (([1:maxAttempts maxAttempts])*finalWaitingTimesSUCRe(:,indProb,1))/(sum(finalWaitingTimesSUCRe(:,indProb,1))+eps);
    meanWaitingTimeSUCRe(indProb,2) = (([1:maxAttempts maxAttempts])*finalWaitingTimesSUCRe(:,indProb,2))/(sum(finalWaitingTimesSUCRe(:,indProb,2))+eps);
    meanWaitingTimeBaseline(indProb) = (([1:maxAttempts maxAttempts])*finalWaitingTimesBaseline(:,indProb))/(sum(finalWaitingTimesBaseline(:,indProb))+eps);    
%     finalWaitingTimesSUCRemeanWaitingTimeSUCRe(indProb,1) = (([1:maxAttempts maxAttempts])*finalWaitingTimesSUCRe(:,indProb,1))/(sum(finalWaitingTimesSUCRe(:,indProb,1)));
%     meanWaitingTimeSUCRe(indProb,2) = (([1:maxAttempts maxAttempts])*finalWaitingTimesSUCRe(:,indProb,2))/(sum(finalWaitingTimesSUCRe(:,indProb,2)));
%     meanWaitingTimeBaseline(indProb) = (([1:maxAttempts maxAttempts])*finalWaitingTimesBaseline(:,indProb))/(sum(finalWaitingTimesBaseline(:,indProb)));    
    
end
%%Plot simulation results
LineWidth =2;
figure;hold on; box on;
plot(K0values,meanWaitingTimeSUCRe(:,1),'r','LineWidth',LineWidth);
plot(K0values,meanWaitingTimeSUCRe(:,2),'b--','LineWidth',LineWidth);
plot(K0values,meanWaitingTimeBaseline,'k-.','LineWidth',LineWidth);

xlabel('Number of Inactive UEs');
ylabel('Average Number of Access Attempts');
legend('SUCRe: No interf.','SUCRe: Interference','Baseline','Location','NorthWest');
ylim([0 10]);


figure;
hold on; box on;

plot(K0values,finalWaitingTimesSUCRe(end,:,1)./sum(finalWaitingTimesSUCRe(:,:,1),1),'r','LineWidth',LineWidth);
plot(K0values,finalWaitingTimesSUCRe(end,:,2)./sum(finalWaitingTimesSUCRe(:,:,2),1),'b--','LineWidth',LineWidth);
plot(K0values,finalWaitingTimesBaseline(end,:)./sum(finalWaitingTimesBaseline,1),'k-.','LineWidth',LineWidth);

xlabel('Number of Inactive UEs');
ylabel('Fraction of Failed Access Attempts');
legend('SUCRe: No interf.','SUCRe: Interference','Baseline','Location','NorthWest');
