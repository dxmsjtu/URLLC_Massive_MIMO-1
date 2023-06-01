%This Matlab script can be used to generate Figure 7, in the article:
%%Emil Bjornson, Elisabeth de Carvalho, Jesper H. Sorensen, Erik G. Larsson,
%Petar Popovski, "A Random Access Protocol for Pilot Allocation in Crowded
%Massive MIMO Systems," IEEE Transactions on Wireless Communications,To appear.
%Download article: http://arxiv.org/pdf/1604.04248
% This is version 1.0 (Last edited: 2017-02-03)
%%License: This code is licensed under the GPLv2 license. If you in any way
%use this code for research that results in publications, please cite our
%original article listed above. 
%Initialization
close all; clear all;
% dd=[1 1 1 0 0 0
% 1 0 0 1 1 0
% 0 1 0 1 0 1
% 0 0 1 0 1 1];
% index =[1 3 4];
% dd(:,index)'*dd(:,index)
%Number of inactive UEs in the cell
K0 = 50000;  K0_range =[5000:2000:5000]*2;
%Probability that a UE wants to become active in a block
pA = 0.005; % pA = 0.05  明显图不合理，看看是代码哪里有的问题，还是越界了
pA_range=[0.001  ]
pA_range*K0_range
%Probability of sending an RA pilot if a UE wants to become active
pP = 1;
%Number of RA pilot signals
taup = 10;
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
        %Compute probability according to the binomial distribution in Eq.(1) and (2) of [1]
        %by using logarithms of each term to gain numerical stability
        probabilities(kind) =  exp(gammaln(K0+1) - gammaln(userValues(kind)+1)- gammaln(K0-userValues(kind)+1) +....,
            (userValues(kind)).*log(pA*pP/taup) + (K0-userValues(kind)).*log((1-pA*pP/taup)));
        probabilities_matrix(kind,caseIndex) =  exp(gammaln(K0+1) - gammaln(userValues(kind)+1)- gammaln(K0-userValues(kind)+1) +....,
            (userValues(kind)).*log(pA*pP/taup) + (K0-userValues(kind)).*log((1-pA*pP/taup)));
        probabilities1(kind) = 1-(1- pA*pP/taup).^(K0) - (K0*pA*pP/taup).*(1-pA*pP/taup).^(K0-userValues(kind));
    end
end
%%Plot simulation results
sum(probabilities(1:3))
figure; box on; % equ (44) of [1] beyes theorem 得到
bar(userValues(2:end),probabilities(2:end)/(1-probabilities(1)));
xlabel('Number of UEs per RA pilot'); ylabel('Probability');
ylim([0 0.3]); ylim([0 max(max(probabilities(2:end)))*1.2]);
figure;
for i =1:max(size(K0_range))
    subplot(max(size(K0_range)),1,i); box on;
    bar(userValues(2:end),probabilities_matrix(2:end,i)/(1-probabilities_matrix(i,1))); 
%The number of UEs, |St|, is distributed as illustrated in Fig. 7, which is obtained from the binomial distribution 
% in (1) by conditioning on that |St| ≥ 1.    
    xlabel('Number of UEs per RA pilot'); ylabel('Probability');
    ylim([0 0.3]); ylim([0 max(max(probabilities_matrix(2:end,1)))*1.2]);
    title_str = strcat('K0 =',num2str(K0_range(i)),',激活率=',num2str(pA),',Number of pilots = ',num2str(taup));
    title(title_str);set(gca,'Fontname','Monospaced');
end
probabilities_matrix=[];
end
h  =gcf; % 获得当前figure 句柄，大家需要用这个模板来画图，仔细调整写出的文章才
MarkerSize=9; LineMethod =1; PlotMethod =1; FontSize=22; YLabelFontSize =22;
LineWidth = 0.5;TitleFontSize = 20; LegendFontSize = 24; axis_ratio=1.5; %myboldify(h,FontSize,LineWidth,LegendFontSize,TitleFontSize);
myboldify(h,MarkerSize,YLabelFontSize,FontSize,LineWidth,LegendFontSize,TitleFontSize)

%%

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
        if strcmp(chtype(1:end), 'text')
            set(hc(j), 'FontSize', LegendFontSize); % 14 pt descriptive labels
        elseif strcmp(chtype(1:end), 'line')
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




