    %This Matlab script can be used to generate Figure 7, in the article:
%%Emil Bjornson, Elisabeth de Carvalho, Jesper H. Sorensen, Erik G. Larsson,
%Petar Popovski, "A Random Access Protocol for Pilot Allocation in Crowded
%Massive MIMO Systems," IEEE Transactions on Wireless Communications,To appear.
%Download article: http://arxiv.org/pdf/1604.04248
%
%This is version 1.0 (Last edited: 2017-02-03)
%
%License: This code is licensed under the GPLv2 license. If you in any way
%use this code for research that results in publications, please cite our
%original article listed above.
%Initialization
close all; clear all;
%Number of inactive UEs in the cell
%Number of inactive UEs in the cell
K0 = 50000;
K0_range =[5000:20000:50000]
%Probability that a UE wants to become active in a block
pA = 0.05; % pA = 0.05  明显图不合理，看看是代码哪里有的问题，还是越界了
pA_range=[ 0.005 ]
%Probability of sending an RA pilot if a UE wants to become active
pP = 1;
%Number of RA pilot signals
taup = 30;
%Range of number of colliding users per RA pilot
userValues = 0:10;
%Define vector to store probabilities
probabilities = zeros(length(userValues),1);

%Go through all user numbers
caseIndex =0;
for pA = pA_range
    probabilities = zeros(length(userValues),1);
caseIndex =0;
for K0 = K0_range
    caseIndex =caseIndex+1;
    for kind = 1:length(userValues)
        %Compute probability according to the binomial distribution in Eq. (1),
        %by using logarithms of each term to gain numerical stability
        probabilities(kind) =  exp(gammaln(K0+1) - gammaln(userValues(kind)+1)- gammaln(K0-userValues(kind)+1) +....,
            (userValues(kind)).*log(pA*pP/taup) + (K0-userValues(kind)).*log((1-pA*pP/taup)));
        probabilities_matrix(kind,caseIndex) =  exp(gammaln(K0+1) - gammaln(userValues(kind)+1)- gammaln(K0-userValues(kind)+1) +....,
            (userValues(kind)).*log(pA*pP/taup) + (K0-userValues(kind)).*log((1-pA*pP/taup)));
        probabilities1(kind) = 1-(1- pA*pP/taup).^(K0) - (K0*pA*pP/taup).*(1-pA*pP/taup).^(K0-userValues(kind));
    end
end
%%Plot simulation results
% figure; box on;
% bar(userValues(2:end),probabilities(2:end)/(1-probabilities(1)));
% xlabel('Number of UEs per RA pilot'); ylabel('Probability');
% ylim([0 0.3]); ylim([0 max(max(probabilities(2:end)))*1.2]);
figure;
for i =1:max(size(K0_range))
    subplot(max(size(K0_range)),1,i); box on;
    bar(userValues(2:end),probabilities_matrix(2:end,i)/(1-probabilities_matrix(1,1)));
    xlabel('Number of UEs per RA pilot'); ylabel('Probability');
    ylim([0 0.3]); ylim([0 max(max(probabilities_matrix(2:end,1)))*1.2]);
    title_str = strcat('K0 =',num2str(K0_range(i)),',激活率=',num2str(pA),',Number of pilots = ',num2str(taup));
    title(title_str);set(gca,'Fontname','Monospaced');
end
probabilities_matrix=[];
end
