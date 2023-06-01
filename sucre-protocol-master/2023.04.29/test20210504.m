% function [meanWaitingTime_Matrix]= Distribution(K0values,nbrOfRAblocks)
clear all;
M_range = [200:100:200]; taup_range = [10:10:10]; pA_range = [1:1:1]/1000;interference_range = [0];method_range = [1 2];
K0values = [100 200 300 500 1000:2000:12000];Default = 0; nbrOfRealizations = 1000; RandomSeed = 0; SumIndex = 0;
tic
for M = M_range for taup = taup_range  for pA = pA_range for interference = interference_range for method = method_range
if RandomSeed == 0; rand('state',12345);  randn('state',12345*3);  end% 保证每次SNR循环，初始种子都一样
para.M = M; para.taup = taup; para.pA = pA; para.interference = interference; para.method = method;
para.K0values = K0values; para.Default = Default;
P_success = RAsim(nbrOfRealizations,para);
toc
SumIndex = SumIndex+1;
if SumIndex ==1
    DataMatrix = P_success;
else
    DataMatrix(size(DataMatrix,1)+1: size(DataMatrix,1)+size(P_success,1),:) =P_success;
end
end;end;end;end;end
DataMatrix
figure;
hold on; box on;
plot_snr_bler(K0values,DataMatrix);ylim([0 1]);
xlabel('Number of Inactive UEs'); ylabel('Success Probability');title('Sim')
legend('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30');
h  =gcf; 
MarkerSize=9; YLabelFontSize =24; FontSize =YLabelFontSize; LineWidth =1.5; LegendFontSize =YLabelFontSize; TitleFontSize =YLabelFontSize;
myboldify(h,MarkerSize,YLabelFontSize,FontSize,LineWidth,LegendFontSize,TitleFontSize)
%%
function P_success = RAsim(nbrOfRealizations,para)
M = para.M; taup = para.taup; pA = para.pA; interference = para.interference;method = para.method;
Default = para.Default;K0values = para.K0values;
if Default==1
M = 100; method = 1;  taup = 10;interference = 0;pA = 1/1000;
end
rho = 1; q = 1; sigma2 = 1; 
maxAttempts = 10;tryAgainProb = 0.5;
Kneighboringcells = 10;shadowFadingStddB = 10;cellRadius = 250;
neighboringBSs = 2*cellRadius*sqrt(3)/2 * exp(1i*(0:5)*pi/3);
userLocationsNeighboring = generatePointsHexagon([Kneighboringcells,nbrOfRealizations,length(neighboringBSs)],cellRadius,0.1*cellRadius);
shadowFadingRealizationsWithinOwnCellUplink = randn([Kneighboringcells,nbrOfRealizations,length(neighboringBSs)]);
shadowFadingRealizationsIntercellUplink = randn([Kneighboringcells,nbrOfRealizations,length(neighboringBSs)]);
for k = 1:Kneighboringcells
    notReady = 1;
    while sum(notReady)>0
        notReady = zeros(length(neighboringBSs),nbrOfRealizations);
        for j = 1:length(neighboringBSs)
            %Check which of the users that are served by the right BS
            notReady(j,:) = (shadowFadingStddB*shadowFadingRealizationsWithinOwnCellUplink(k,:,j) - 38*log10(abs(userLocationsNeighboring(k,:,j))) < shadowFadingStddB*shadowFadingRealizationsIntercellUplink(k,:,j) - 38*log10(abs(userLocationsNeighboring(k,:,j)-neighboringBSs(j))) );
            shadowFadingRealizationsWithinOwnCellUplink(k,notReady(j,:)>0,j) = randn(1,sum(notReady(j,:)>0));
            shadowFadingRealizationsIntercellUplink(k,notReady(j,:)>0,j) = randn(1,sum(notReady(j,:)>0));
        end
    end
end
if interference == 1; qNeighbor = taup*q; rhoIntercell = rho;end
if interference == 0; qNeighbor = 0;  rhoIntercell = 0;   end
interCellVarianceUplink = zeros(1,nbrOfRealizations);
for j = 1:length(neighboringBSs)
    interCellVarianceUplink = interCellVarianceUplink + sum(rhoIntercell*10.^( (27+ 98.65 - 34.53 - 38*log10(abs(userLocationsNeighboring(:,:,j) + neighboringBSs(j))) + shadowFadingStddB*shadowFadingRealizationsIntercellUplink(:,:,j)  )/10   ),1) ;
end
interCellBias = mean(interCellVarianceUplink);
% finalWaitingTimes = zeros(maxAttempts+1,length(K0values));
% meanWaitingTime = zeros(1,length(K0values));
%Go through all different number of inactive UEs
for indProb = 1:length(K0values)
%     disp(['K0 values: ' num2str(indProb) ' out of ' num2str(length(K0values))])
    K0 = K0values(indProb);
    newUsers = binornd(K0,pA,[nbrOfRealizations 1]);
    if method ==1;   Pilot_combination = (1:taup)';   end
    if method ==2;    Pilot_combination = nchoosek(1:taup,2);  end ;warning('off');
    waitingTime = []; %Contains number of access attempts
    waitingBetas = []; %Channel variance of the users
    waitingIntercellInter = []; %Downlink inter-cell inteference variance
    successNum = 0;
for r = 1:nbrOfRealizations
    newUserLocations = generatePointsHexagon([newUsers(r) 1],cellRadius,0.1*cellRadius);
    newUserDistance = abs(newUserLocations);
    newShadowFading = randn(newUsers(r),1);
    shadowFadingRealizationsIntercellDownlink = randn(newUsers(r),length(neighboringBSs));
    notReady = 1;
    while sum(notReady)>0
        notReady = zeros(newUsers(r),1);
        for j = 1:length(neighboringBSs)
            notReady = notReady + (shadowFadingStddB*newShadowFading - 38*log10(newUserDistance) < shadowFadingStddB*shadowFadingRealizationsIntercellDownlink(:,j) - 38*log10(abs(newUserLocations-neighboringBSs(j))) );
        end
        newShadowFading(notReady>0) = randn(sum(notReady>0),1);
        shadowFadingRealizationsIntercellDownlink(notReady>0,:) = randn(sum(notReady>0),length(neighboringBSs));
    end
    newBetas = 10.^( (27+ 98.65 - 34.53 - 38*log10(newUserDistance) + shadowFadingStddB*newShadowFading  )/10   );
    newIntercellVarianceDownlink = zeros(newUsers(r),1);
    for j = 1:length(neighboringBSs)
        newIntercellVarianceDownlink = newIntercellVarianceDownlink + qNeighbor*10.^( (27+ 98.65 - 34.53 - 38*log10(abs(newUserLocations-neighboringBSs(j))) + shadowFadingStddB*shadowFadingRealizationsIntercellDownlink(:,j)  )/10   );
    end
    interCellVarianceDownlink = [newIntercellVarianceDownlink; waitingIntercellInter];
    betas = [newBetas;waitingBetas];
    numberOfAccessingUsers = length(betas);
    shouldWaitingUsersRetransmit = binornd(1,tryAgainProb,size(waitingTime));
    accessAttempt = [ones(newUsers(r),1) ; shouldWaitingUsersRetransmit];
    % 选择导频
    if method == 1
        pilotSelections = zeros(numberOfAccessingUsers,1);
        for user_n = 1:numberOfAccessingUsers
            index = randperm(size(Pilot_combination,1),1); pilotSelections(user_n,:) = Pilot_combination(index,:);
        end
%         pilotSelections = accessAttempt.*pilotSelections;
    end
    if method == 2 
        pilotSelections = zeros(numberOfAccessingUsers,2);
        for user_n = 1:numberOfAccessingUsers
            index = randperm(size(Pilot_combination,1),1); pilotSelections(user_n,:) = Pilot_combination(index,:);
        end
%         pilotSelections = repmat(accessAttempt,1,2).*pilotSelections;
    end
%     pilotSelections = accessAttempt.*pilotSelections;
    accessAttempts = [ones(newUsers(r),1); waitingTime+shouldWaitingUsersRetransmit];
    successfulAttempt = false(size(betas));successfulAttempt_1 = successfulAttempt ;successfulAttempt_2 = successfulAttempt ;
    if ~isempty(accessAttempts)
        h = (randn(M,numberOfAccessingUsers)+1i*randn(M,numberOfAccessingUsers));
        h = repmat(sqrt(betas'/2),[M 1]) .* h;
        if interference == 0
            noiseInterfvariance = sigma2;
            eta = sqrt(sigma2/2)*(randn(1,numberOfAccessingUsers)+1i*randn(1,numberOfAccessingUsers)); 
        end
        if interference == 1
            noiseInterfvariance = sigma2+interCellVarianceUplink(1,r);
            eta = sqrt((sigma2+interCellVarianceDownlink')/2).*(randn(1,numberOfAccessingUsers)+1i*randn(1,numberOfAccessingUsers));
        end
        
        if method ==1 %原始单码分布式接入
        for t = 1:taup
            userIndices = find(pilotSelections(:,1)==t);
            n = sqrt(1/2)*(randn(M,1)+1i*randn(M,1));
            yt = sqrt(taup*rho) * sum(h(:,userIndices),2) + sqrt(noiseInterfvariance)*n;
            v = sqrt(q)*yt/norm(yt);
            retransmit = false(length(userIndices),1);
            for k = 1:length(userIndices)
                z = sqrt(taup)*sum(conj(h(:,userIndices(k))).*v,1) + eta(1,userIndices(k));
                alphaEst_approx2 = exp(gammaln(M+1/2)-gammaln(M))^2*q*taup^2*rho*(betas(userIndices(k))./real(z)).^2-sigma2;
                if alphaEst_approx2<rho*betas(userIndices(k))*taup; alphaEst_approx2 = rho*betas(userIndices(k))*taup; end
                retransmit(k,1) =  rho*betas(userIndices(k))*taup>(alphaEst_approx2-interCellBias)/2-betas(userIndices(k))/sqrt(M);
            end
            if sum(retransmit) == 1;  successfulAttempt(userIndices)=retransmit;   end
        end
        end        

        if method == 2%分时隙双码
            for t_1 = 1:taup
                userIndices = find(pilotSelections(:,1)==t_1);
                n = sqrt(1/2)*(randn(M,1)+1i*randn(M,1));
                yt = sqrt(taup/2*rho) * sum(h(:,userIndices),2) + sqrt(noiseInterfvariance/2)*n;
                v = sqrt(q)*yt/norm(yt);
                retransmit = false(length(userIndices),1);
                for k = 1:length(userIndices)
                    z = sqrt(taup/2)*sum(conj(h(:,userIndices(k))).*v,1) + eta(1,userIndices(k));
                    alphaEst_approx2 = exp(gammaln(M+1/2)-gammaln(M))^2*q*(taup/2)^2*rho*(betas(userIndices(k))./real(z)).^2-sigma2;
                    if alphaEst_approx2<rho*betas(userIndices(k))*taup/2; alphaEst_approx2 = rho*betas(userIndices(k))*taup/2; end
                    retransmit(k,1) =  rho*betas(userIndices(k))*taup/2>(alphaEst_approx2-interCellBias)/2-betas(userIndices(k))/sqrt(M);
                end
                if sum(retransmit) == 1
                    successfulAttempt_1(userIndices)=retransmit;
                end
            end
            for t_2 = 1:taup
                userIndices = find(pilotSelections(:,2)==t_2);
                n = sqrt(1/2)*(randn(M,1)+1i*randn(M,1));
                yt = sqrt(taup/2*rho) * sum(h(:,userIndices),2) + sqrt(noiseInterfvariance/2)*n;
                v = sqrt(q)*yt/norm(yt);
                retransmit = false(length(userIndices),1);
                for k = 1:length(userIndices)
                    z = sqrt(taup/2)*sum(conj(h(:,userIndices(k))).*v,1) + eta(1,userIndices(k));
                    alphaEst_approx2 = exp(gammaln(M+1/2)-gammaln(M))^2*q*(taup/2)^2*rho*(betas(userIndices(k))./real(z)).^2-sigma2;
                    if alphaEst_approx2<rho*betas(userIndices(k))*taup/2; alphaEst_approx2 = rho*betas(userIndices(k))*taup/2; end
                    retransmit(k,1) =  rho*betas(userIndices(k))*taup/2>(alphaEst_approx2-interCellBias)/2-betas(userIndices(k))/sqrt(M);
                end
                if sum(retransmit) == 1
                    successfulAttempt_2(userIndices)=retransmit;
                end
            end
            successfulAttempt = successfulAttempt_1 | successfulAttempt_2;
        end   
    end
    if ~isempty(successfulAttempt) 
        if successfulAttempt(1) == 1; successNum = successNum+1; end %统计第一个用户成功接入的次数 
    end
    AccessUserNum(r,1) = length(find(successfulAttempt==1));
%     %% theory
%     for user_n = 1:numberOfAccessingUsers-1
%         S1(user_n) = exp(gammaln(numberOfAccessingUsers)-gammaln(user_n+1)-gammaln(numberOfAccessingUsers-1-user_n+1)...
%                        +(numberOfAccessingUsers-1-user_n).*log(taup-1));
%         S2(user_n) = user_n /(user_n +1);
%     end
%     stop = 1;
%     S(r,1) = sum(S1.*S2)/(taup^numberOfAccessingUsers);
end
    P_successOfSingleUE(indProb) = successNum/nbrOfRealizations;% 单个用户的成功接入概率  
    P_successOfAllUE(indProb) = sum(AccessUserNum)/sum(newUsers);% 系统成功接入概率
end
 STOP =1;
 P_success = P_successOfSingleUE; % 返回单个用户的成功概率  
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
function myboldify(h,MarkerSize,YLabelFontSize,FontSize,LineWidth,LegendFontSize,TitleFontSize)
%h  =gcf; % 获得当前figure 句柄，大家需要用这个模板来画图，仔细调整写出的文章才
%FontSize= 24 ; LineWidth = 3;TitleFontSize = 20; LegendFontSize = 36; axis_ratio=1.5; %myboldify(h,FontSize,LineWidth,LegendFontSize,TitleFontSize);
%  myboldify(h,FontSize,LineWidth,LegendFontSize,TitleFontSize)
% myboldify: make lines and text bold;  boldifies the current figure; applies to the figure with the handle h
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
%         set(get(ha(i), 'YLabel'), 'MarkerSize', MarkerSize));
%         set(get(ha(i), 'MarkerSize', MarkerSize);
    end
    hc = get(ha(i), 'Children'); % the objects within an axis
    
    for j = 1:length(hc)
        chtype = get(hc(j), 'Type');
        if strcmp(chtype(1:length(chtype)), 'text')
            set(hc(j), 'FontSize', LegendFontSize); % 14 pt descriptive labels
            set(hc(j), 'LineWidth', LineWidth);
            set(hc(j), 'MarkerSize', MarkerSize);
        elseif strcmp(chtype(1:length(chtype)), 'line')
            
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
