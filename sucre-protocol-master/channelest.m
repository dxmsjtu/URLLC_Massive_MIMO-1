clear all; close all;
%  qpsk,3/4 EVA,
SNR_Matrix=[  6     8    10    12    14
     6     8    10    12    14
     6     8    10    12    14
     6     8    10    12    14     ];
 BER_Matrix=[
0.0240320910973085	0.0105810041407868	0.00357315389924086	0.00112318840579710	0.000420289855072464
0.0223276397515528	0.0084665631469979	0.0024639751552795	0.0006821946169772	0.000159109730849
0.0218814699792961	0.00789042443064182	0.00205314009661836	0.00050356107660455	0.000113354037267081
0.0174715320910973	0.00620082815734990	0.00137249827467219	0.000293271221532091	6.1542028985507e-05]

figure(1)
plot_snr_bler(SNR_Matrix,BER_Matrix) 
axis([min(min(SNR_Matrix))*1.0 max(max(SNR_Matrix))*1.0 min(min(BER_Matrix))*1.0 max(max(BER_Matrix))*1.0]);

h  =gcf;
MarkerSize=9; YLabelFontSize =10; FontSize =10; 
LineWidth =1.5; LegendFontSize =10; TitleFontSize =10;
MarkerSize=18; YLabelFontSize =22; FontSize =22; 
LineWidth =3; LegendFontSize =22; TitleFontSize =22;


myboldify(h,MarkerSize,YLabelFontSize,FontSize,LineWidth,LegendFontSize,TitleFontSize)
legend('Mathworks','USTB method1','USTB method 2')
legend('USTB','M_squeeze','random sequence','USTB method 2')
legend('Proposed','M-squeeze','Perfect')
ylabel('BER');
xlabel('SNR(dB)');
 title('EVA，1*2MIMO  6PRB  QPSK, rate =3/4， FFT=2048')
  legend('DFT','2倍对称DFT',' 2^N-DFT','perfect')
   legend('DFT','改进DFT','改进DFT(北京科技大学)','perfect') 
% title('QPSK, rate = 3/4, 发送数据帧结构: 512导频+256cp+512数据') 
 set(gca,'Fontname','Monospaced'); 
% figure(2)
% plot_snr_bler(SNR_Matrix,CANNEL_MSE_Matrix)
% axis([min(min(SNR_Matrix))*1.0 max(max(SNR_Matrix))*1.0 min(min(CANNEL_MSE_Matrix))*1.0 max(max(CANNEL_MSE_Matrix))*1.0]);
% 
% h  =gcf;
% MarkerSize=9; YLabelFontSize =10; FontSize =10; 
% LineWidth =1.5; LegendFontSize =10; TitleFontSize =10;
% myboldify(h,MarkerSize,YLabelFontSize,FontSize,LineWidth,LegendFontSize,TitleFontSize)
% legend('Mathworks','USTB method1','USTB method 2')
% legend('USTB','M_squeeze','random sequence','USTB method 2')
% legend('Proposed','M-squeeze','Perfect')
% title('QPSK, rate = 1/2, 发送数据帧结构: 512导频+256cp+512数据') 
%  set(gca,'Fontname','Monospaced'); 
% ylabel('NMSE');
% xlabel('SNR(dB)');

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