clear all; close all;
filenameTmp ='BER_N_K';%可将仿真关键参数作为文件名
mkdir_str=strcat('.\Simulation\',filenameTmp);
mkdir(mkdir_str);%一运行就会在当前文件夹下创建simulation文件夹
mkdir_str1 =strcat(mkdir_str,'\');mkdir_str =strcat(mkdir_str1,filenameTmp);
mkdir_str =strcat(mkdir_str,'.m'); Save_link1=Save_link_file('test.m',mkdir_str);%将当前函数另存为nk文件
%%%两个仿真参数%%%
N=[10 20]; K=[2 4];
ber_index=0;%不同的NK组合一共有4种，所以BER_matrix共4行，由ber_index设置
for n=N   
    for k=K
        ber_index=ber_index+1;
        for snr_num=1:5
            snr=snr_num*2;
            data=randi([0 1],n,k);
            data_hat=randi([0 1],n,k);
            ber(snr_num)=sum(sum(abs(data_hat-data)))/(n*k);
        end
        BER_matrix(ber_index,:)=ber;%就算出当前NK状态下的BER，并存在BER_matrix的第ber_index行        
      %%    simulation文件夹里，保存每次仿真结果  ,成  .mat文件
        strsave= strcat('.\Simulation\',filenameTmp,'\');
        if ber_index ==1
            filenameTmp1 = strcat(filenameTmp,'.mat');
        end
        strsave= strcat(strsave,filenameTmp1);
        s=['save ' strsave];% 保持.mat 文件，以后仿真结果可以再次确认,以后一定注意可以再次画图。
        eval(s);
    end
end

loadFile =;
if loadFile ==1    
    filenameTmp='BER_N_K'; %MUSAvs_USTB_4_20_EPA MUSAvs_USTB_4_16_EPA
    loadStr = strcat('.\Simulation\',filenameTmp);    loadStr1 = strcat(loadStr,'\');
    loadStr = strcat(loadStr1,filenameTmp);    loadStr = strcat(loadStr,'.mat');
    loadStr = mystrcat('load ',loadStr);
    eval(loadStr);
    BLER_Matrix
    seltected =[1:size(BLER_Matrix,1)];
    seltected=[ 10 13 16];
    BLER_Matrix1 =BLER_Matrix(seltected,:);
    SNR_Matrix1 =SNR_Matrix(seltected,:);
    plot_snr_bler(SNR_Matrix1,BLER_Matrix1);
    plot_snr_bler(SNR_Matrix,BLER_Matrix);%
    legend('Code-Collision', 'No-Collision','proposed, Localized Mapping','MUSA, Distributed Mapping','proposed, Distributed Mapping','MUSA, Distributed Mapping','proposed, Distributed Mapping')
     title('  TDL-C  1Tx2Rx  6PRB  QPSK  12users,rate =0.5 \bf{MMSE-SIC} ')
     title('  EVA  1Tx2Rx  6PRB  QPSK  16users,\bf{rate =0.6} ')
     title('TDL-C  1Tx2Rx  16PRB  16QAM  12users, \bf{ rate =0.25} ')
    ylabel('BLER'); xlabel('SNR (dB)');
    
    legend('MUSA-No-collision','MUSA-collision','4168-prb','4168-fre')
     legend('PDMA(462)-No-collision','PDMA(462)-collision','4168-prb','4168-fre')
      legend('MUSA','USTB','4168-prb','4168-fre')
    h  =gcf; % 获得当前figure 句柄，你们可以调整 FontSize= 24 ; LineWidth = 3;TitleFontSize = 20; LegendFontSize = 36;来获得最优效果
    MarkerSize=12; YLabelFontSize =40; FontSize =40; LineWidth =4; LegendFontSize =40; TitleFontSize =40;
%     MarkerSize=12; YLabelFontSize =25; FontSize =25; LineWidth =2; LegendFontSize =25; TitleFontSize =25;
    myboldify(h,MarkerSize,YLabelFontSize,FontSize,LineWidth,LegendFontSize,TitleFontSize)
    axis([min(min(SNR_Matrix))*1.0 max(max(SNR_Matrix))*1.0 min(min(BLER_Matrix))*1.0 max(max(BLER_Matrix))*1.0]);
end
end


SNR_Matrix=[  6     8    10    12    14
    6     8    10    12    14
    6     8    10    12    14
    6     8    10    12    14     ];
BER_Matrix=[
    0.0240320910973085	0.0105810041407868	0.00357315389924086	0.00112318840579710	0.000420289855072464
    0.0223276397515528	0.0084665631469979	0.0024639751552795	0.0006821946169772	0.000159109730849
    0.0218814699792961	0.00789042443064182	0.00205314009661836	0.00050356107660455	0.000113354037267081
    0.0174715320910973	0.00620082815734990	0.00137249827467219	0.000243271221532091	6.1542028985507e-05]
figure(1)
plot_snr_bler(SNR_Matrix,BER_Matrix)
axis([min(min(SNR_Matrix))*1.0 max(max(SNR_Matrix))*1.0 min(min(BER_Matrix))*1.0 max(max(BER_Matrix))*1.0]);
h  =gcf;
MarkerSize=9; YLabelFontSize =10; FontSize =10;
LineWidth =1.5; LegendFontSize =10; TitleFontSize =10;
MarkerSize=18; YLabelFontSize =22; FontSize =22;
LineWidth =3; LegendFontSize =22; TitleFontSize =22;

myboldify(h,MarkerSize,YLabelFontSize,FontSize,LineWidth,LegendFontSize,TitleFontSize)
ylabel('BER');
xlabel('SNR(dB)');
title('EVA，1*2MIMO  6PRB  QPSK, rate =3/4， FFT=2048')
legend('DFT','Method1 ','Method2','perfect')
set(gca,'Fontname','Monospaced');

function  plot_snr_bler(SNR_Matrix,BLER_Matrix)
LineStyles='-bs -gs -rp -bv -gv -rv ';
Rows=size(BLER_Matrix,1);
LineStyles=parse(LineStyles);
MarkerSize =9;
LineWidth = 1.5;
for i=1:Rows
    semilogy(SNR_Matrix(i,:),BLER_Matrix(i,:),LineStyles(i,:),'LineWidth',LineWidth,'MarkerSize',MarkerSize);
    hold on;
    grid on;
end
end

function [x] = parse(inStr)
sz=size(inStr);
strLen=sz(2);
x=blanks(strLen);
%x=blanks(strLengthMax);2002/5/12 modify
wordCount=1;
last=0;
for i=1:strLen
    if inStr(i) == ' '
        wordCount = wordCount + 1;
        x(wordCount,:)=blanks(strLen);
        %x(wordCount,:)=blanks(strLengthMax); 2002/5/12 modify
        last=i;
    else
        x(wordCount,i-last)=inStr(i);
    end
end
end

function myboldify(h,MarkerSize,YLabelFontSize,FontSize,LineWidth,LegendFontSize,TitleFontSize)
% myboldify: make lines and text bold
%   myboldify boldifies the current figure
%   myboldify(h) applies to the figure with the handle h

if nargin < 1
    h = gcf;
end

ha = get(h, 'Children'); % the handle of each axis

for i = 1:length(ha)
    
    if strcmp(get(ha(i),'Type'), 'axes') % axis format
        set(ha(i), 'FontSize', FontSize);      % tick mark and frame format
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