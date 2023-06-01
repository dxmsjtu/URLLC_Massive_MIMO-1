function [meanWaitingTime_Matrix]= Distribution(K0values,nbrOfRAblocks)
%Initialization
%Set the number of Monte-Carlo realizations
nbrOfRAblocks =100;%Number of BS antennas
M = 100;
K0values = [100:100:900 1000:2000:21000];
% K0values = [100 250 500:1000:12000];%Range of number of inactive UEs in the cell
pA = 0.001;%Probability that an inactive UE wants to become active in a given block
maxAttempts = 10;
tryAgainProb = 0.5;
taup = 10;
rho = 1; %Transmit power of UEs in the cell
q = 1; %Transmit power of the BS
sigma2 = 1; %Noise variance
%Standard deviation of shadow fading
shadowFadingStddB = 10;

%Set cell radius (in meter)
cellRadius = 250;


%Generate noise realizations at the BS in the center cell
% n = sqrt(1/2)*(randn(M,taup,nbrOfRAblocks)+1i*randn(M,taup,nbrOfRAblocks));
Pilot_Pool = [nchoosek([1:taup],2);-nchoosek([-taup:-1],2)];
%                 for i = 1:size(Pilot_Pool,1)
n = sqrt(1/2)*(randn(M,taup,nbrOfRAblocks)+1i*randn(M,taup,nbrOfRAblocks));


%Set number of active UEs in the neighboring cells
Kneighboringcells = 10;

%Set transmit power of users in adjacent cells
rhoIntercell = 1;


%Compute locations of all neighboring BSs
neighboringBSs = 2*cellRadius*sqrt(3)/2 * exp(1i*(0:5)*pi/3);

%Generate user locations in neighboring cells
userLocationsNeighboring = generatePointsHexagon([Kneighboringcells,nbrOfRAblocks,length(neighboringBSs)],cellRadius,0.1*cellRadius);

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
            
            %Check which of the users that are served by the right BS
            notReady(j,:) = (shadowFadingStddB*shadowFadingRealizationsWithinOwnCellUplink(k,:,j) - 38*log10(abs(userLocationsNeighboring(k,:,j))) < shadowFadingStddB*shadowFadingRealizationsIntercellUplink(k,:,j) - 38*log10(abs(userLocationsNeighboring(k,:,j)-neighboringBSs(j))) );
            
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
    interCellVarianceUplink = interCellVarianceUplink + sum(rhoIntercell*10.^( (27+ 98.65 - 34.53 - 38*log10(abs(userLocationsNeighboring(:,:,j) + neighboringBSs(j))) + shadowFadingStddB*shadowFadingRealizationsIntercellUplink(:,:,j)  )/10   ),1) ;
    
end


%Compute the average uplink inter-cell interference, which is used in the
%bias terms
interCellBias = mean(interCellVarianceUplink);
%There are three methods that are considered:
%1: SUCRe without inter-cell interference
%2: SUCRe with inter-cell interference
method_range = [1];
L_range = [1 2 3]; % Ê±Ï¶ÊýÄ¿
%Matrices to store the simulation results, in terms of waiting time for the
%SUCRe protocol (without or with inter-cell interference) and baseline scheme
finalWaitingTimesSUCRe = zeros(maxAttempts+1,length(K0values),length(method_range));
finalWaitingTimesBaseline = zeros(maxAttempts+1,length(K0values));
meanWaitingTime = zeros(length(K0values),length(method_range));
meanFailProbability = zeros(length(K0values),length(method_range));
case_index = 0;
for L = L_range
    case_index = case_index+1;
%Go through all different number of inactive UEs
for indProb = 1:length(K0values)
    %Display simulation progress
    disp(['K0 values: ' num2str(indProb) ' out of ' num2str(length(K0values))])
    K0 = K0values(indProb);
    newUsers = binornd(K0,pA,[nbrOfRAblocks 1]);
    for method = method_range
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
        %Go through all RA blocks that are considered in the Monte-Carlo
        %simulations
        for r = 1:nbrOfRAblocks
            newUserLocations = generatePointsHexagon([newUsers(r) 1],cellRadius,0.1*cellRadius);
            newUserDistance = abs(newUserLocations);
            newShadowFading = randn(newUsers(r),1);
            shadowFadingRealizationsIntercellDownlink = randn(newUsers(r),length(neighboringBSs));
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
            %(27 dBm represents the transmit power and -98.65 dBm
            %represents the noise variance)
            newBetas = 10.^( (27+ 98.65 - 34.53 - 38*log10(newUserDistance) + shadowFadingStddB*newShadowFading  )/10   );               
            %Compute the total inter-cell interference in the uplink and downlink in
            %each channel realization
            newIntercellVarianceDownlink = zeros(newUsers(r),1);
            for j = 1:length(neighboringBSs)
                
                %Note: 27 dBm represents the transmit power and -98.65 dBm
                %represents the noise variance
                newIntercellVarianceDownlink = newIntercellVarianceDownlink + qNeighbor*10.^( (27+ 98.65 - 34.53 - 38*log10(abs(newUserLocations-neighboringBSs(j))) + shadowFadingStddB*shadowFadingRealizationsIntercellDownlink(:,j)  )/10   );
                
            end
            interCellVarianceDownlink = [newIntercellVarianceDownlink; waitingIntercellInter];
            %Combine the new UEs with the ones that have made previous
            %access attempts
            betas = [newBetas; waitingBetas];
            numberOfAccessingUsers = length(betas);
            shouldWaitingUsersRetransmit = binornd(1,tryAgainProb,size(waitingTime));
            accessAttempt = [ones(newUsers(r),1) ; shouldWaitingUsersRetransmit]; 
            %Randomize which of the pilots that each of the UEs are using
            pilotSelections = zeros([numberOfAccessingUsers L]);
            successfulAccess = zeros([numberOfAccessingUsers L]);
            for index_numberOfAccessingUsers = 1:numberOfAccessingUsers
                pilotSelections(index_numberOfAccessingUsers,:) = randperm(taup,L);
                %     Pilot_user(index_numberOfAccessingUsers,:)=randi([1,Pilot_N],1,Pilot_n);
            end
            pilotSelections = accessAttempt.*pilotSelections;
%               pilotSelections = accessAttempt.*randi(taup,[numberOfAccessingUsers 2]);

            %Count the number of pilots that each of the UEs will have
            %transmitted, after this block
            accessAttempts = [ones(newUsers(r),1); waitingTime+shouldWaitingUsersRetransmit];
                        
            %Check if there is at least on UE that transmit pilots in this
            %RA block
            if ~isempty(accessAttempts)
                %Generate uncorrelated Rayleigh fading channel realizations
                h = (randn(M,numberOfAccessingUsers)+1i*randn(M,numberOfAccessingUsers));
                h = repmat(sqrt(betas'/2),[M 1]) .* h;               
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
                %                 Pilot_Pool = [nchoosek([1:taup],2);-nchoosek([-taup:-1],2)];
                if L == 1 || L == 2 ||  L == 3
                for t_1 = 1:taup
                    userIndices_1 = find(pilotSelections(:,1)==t_1);
                    userIndices = userIndices_1;
                    %Consider the SUCRe protocol
                    if method == 1 || method == 2
                        %Compute the received signal in Eq. (6)
                        yt = sqrt(taup*rho) * sum(h(:,userIndices),2) + sqrt(noiseInterfvariance)*n(:,t_1,r);
                        %Compute the precoding vector used by the BS
                        v = sqrt(q)*yt/norm(yt);
                        %Prepare a list of UEs that decide to retransmit pilot t
                        retransmit_1 = false(length(userIndices),1);
                        %                         retransmit_matrix = zeros([numberOfAccessingUsers 2]);
                        %Go through the UEs that transmitted pilot t
                        for k = 1:length(userIndices)
                            z = sqrt(taup)*sum(conj(h(:,userIndices(k))).*v,1) + eta(1,userIndices(k));
                            alphaEst_approx2 = exp(gammaln(M+1/2)-gammaln(M))^2*q*taup^2*rho*(betas(userIndices(k))./real(z)).^2-sigma2;
                            if alphaEst_approx2<rho*betas(userIndices(k))*taup
                                alphaEst_approx2 = rho*betas(userIndices(k))*taup;
                            end
                            if method == 1
                                retransmit_1(k) =  rho*betas(userIndices(k))*taup>alphaEst_approx2/2-betas(userIndices(k))/sqrt(M);
                            elseif method == 2
                                retransmit_1(k) =  rho*betas(userIndices(k))*taup>(alphaEst_approx2-interCellBias)/2-betas(userIndices(k))/sqrt(M);
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
                    for t_2 = 1:taup
                        userIndices_2 = find(pilotSelections(:,2)==t_2);
                        userIndices = userIndices_2;
                        if method == 1 || method == 2
                            yt = sqrt(taup*rho) * sum(h(:,userIndices),2) + sqrt(noiseInterfvariance)*n(:,t_2,r);
                            v = sqrt(q)*yt/norm(yt);
                            retransmit_2 = false(length(userIndices),1);
                            %Go through the UEs that transmitted pilot t
                            for k = 1:length(userIndices)
                                z = sqrt(taup)*sum(conj(h(:,userIndices(k))).*v,1) + eta(1,userIndices(k));
                                alphaEst_approx2 = exp(gammaln(M+1/2)-gammaln(M))^2*q*taup^2*rho*(betas(userIndices(k))./real(z)).^2-sigma2;
                                if alphaEst_approx2<rho*betas(userIndices(k))*taup
                                    alphaEst_approx2 = rho*betas(userIndices(k))*taup;
                                end
                                if method == 1
                                    retransmit_2(k) =  rho*betas(userIndices(k))*taup>alphaEst_approx2/2-betas(userIndices(k))/sqrt(M);
                                elseif method == 2
                                    retransmit_2(k) =  rho*betas(userIndices(k))*taup>(alphaEst_approx2-interCellBias)/2-betas(userIndices(k))/sqrt(M);
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
                    for t_3 = 1:taup
                        userIndices_3 = find(pilotSelections(:,3)==t_3);
                        userIndices = userIndices_3;
                        if method == 1 || method == 2
                            yt = sqrt(taup*rho) * sum(h(:,userIndices),2) + sqrt(noiseInterfvariance)*n(:,t_3,r);
                            v = sqrt(q)*yt/norm(yt);
                            retransmit_3 = false(length(userIndices),1);
                            %Go through the UEs that transmitted pilot t
                            for k = 1:length(userIndices)
                                z = sqrt(taup)*sum(conj(h(:,userIndices(k))).*v,1) + eta(1,userIndices(k));
                                alphaEst_approx2 = exp(gammaln(M+1/2)-gammaln(M))^2*q*taup^2*rho*(betas(userIndices(k))./real(z)).^2-sigma2;
                                if alphaEst_approx2<rho*betas(userIndices(k))*taup
                                    alphaEst_approx2 = rho*betas(userIndices(k))*taup;
                                end
                                if method == 1
                                    retransmit_3(k) =  rho*betas(userIndices(k))*taup>alphaEst_approx2/2-betas(userIndices(k))/sqrt(M);
                                elseif method == 2
                                    retransmit_3(k) =  rho*betas(userIndices(k))*taup>(alphaEst_approx2-interCellBias)/2-betas(userIndices(k))/sqrt(M);
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
                            finalWaitingTimesSUCRe(accessAttempts(index_1(i)),indProb,method) = finalWaitingTimesSUCRe(accessAttempts(index_1(i)),indProb,method) + 1;
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
                            finalWaitingTimesSUCRe(accessAttempts(index_1_2(i)),indProb,method) = finalWaitingTimesSUCRe(accessAttempts(index_1_2(i)),indProb,method) + 1;
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
                            finalWaitingTimesSUCRe(accessAttempts(index_1_2_3(i)),indProb,method) = finalWaitingTimesSUCRe(accessAttempts(index_1_2_3(i)),indProb,method) + 1;
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
                        finalWaitingTimesSUCRe(end,indProb,method) = finalWaitingTimesSUCRe(end,indProb,method) + sum(giveUp);
                    elseif method == 3
                        finalWaitingTimesBaseline(end,indProb) = finalWaitingTimesBaseline(end,indProb) + sum(giveUp);
                    end
                end
                
                %Keep the important parameters for all the UEs that failed
                %to access the network and has not given up
                waitingTime = accessAttempts((successfulAttempt==false) & (accessAttempts<maxAttempts));
                waitingBetas = betas((successfulAttempt==false) & (accessAttempts<maxAttempts));
                waitingIntercellInter = interCellVarianceDownlink((successfulAttempt==false) & (accessAttempts<maxAttempts));
                stop = 1;
            end
        end
        meanWaitingTime(indProb,method) = (([1:maxAttempts maxAttempts])*finalWaitingTimesSUCRe(:,indProb,method))/sum(finalWaitingTimesSUCRe(:,indProb,method));
%         meanFailProbability(indProb,method) = finalWaitingTimesSUCRe(end,:,method)./sum(finalWaitingTimesSUCRe(:,indProb,method),1);
    end
end
if case_index == 1
    meanWaitingTime_Matrix = meanWaitingTime';
%     meanFailProbability_Matrix = meanFailProbability';
else
    meanWaitingTime_Matrix(size(meanWaitingTime_Matrix,1)+1: size(meanWaitingTime_Matrix,1)+size(meanWaitingTime',1),:) = meanWaitingTime';
%     meanFailProbability_Matrix(size(meanFailProbability_Matrix,1)+1: size(meanFailProbability_Matrix,1)+size(meanFailProbability',1),:) = meanFailProbability';
end
stop = 1;

end
%% plot
figure
hold on; box on;
plot_snr_bler(K0values,meanWaitingTime_Matrix)
set(gca,'Fontname','Monospaced'); set(gca,'FontSize',12);
xlabel('Number of Inactive UEs'); ylabel('Average Number of Access Attempts');
legend('L=1','L=2','L=3','Location','SouthEast');ylim([0 10]);
end
% %%Plot simulation results
% figure;
% hold on; box on;
% plot(K0values,meanWaitingTime_Matrix(1,:),'r--','LineWidth',1);
% plot(K0values,meanWaitingTime_Matrix(2,:),'b--','LineWidth',1);
% plot(K0values,meanWaitingTime_Matrix(3,:),'k--','LineWidth',1);
% set(gca,'Fontname','Monospaced'); set(gca,'FontSize',12);
% xlabel('Number of Inactive UEs');ylabel('Average Number of Access Attempts');
% legend('L=1','L=2','L=3','Location','SouthEast');ylim([0 10]);

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