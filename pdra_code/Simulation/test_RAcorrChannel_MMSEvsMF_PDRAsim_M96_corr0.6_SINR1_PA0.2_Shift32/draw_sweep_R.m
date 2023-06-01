clc;clear all;close all;
a=load('test_RAcorrChannel_MMSEvsMF_PDRAsim_M96_corr0.6_SINR1_PA0.2_Shift32.mat');
%b=load('test_RAcorrChannel_MMSEvsMF_PDRAsim_M256_corr0.6_SINR4_PA0.2_Shift32.mat');
data = [a.DataMatrix];
figure; hold on;
plot_snr_bler(a.R_range,data)
%title('Success Probability')
xlabel('R', 'Interpreter','latex'); ylabel('Success Probability', 'Interpreter','latex');
set(gca, 'YScale', 'log');set(gca,'Fontname','Monospaced'); set(gca,'FontSize',12);set(gca,'XTick',a.R_range);set(gca, 'XMinorGrid','on');
%legend('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30');
legend('MMSE M=96','MF M=96');
%legend('complete collide MF','complete collide MMSE','component collide MF','component collide MMSE');
h  =gcf; % 获得当前figure 句柄，大家需要用这个模板来画图，仔细调整写出的文章才
MarkerSize=9; LineWidth =2; LineMethod =1; PlotMethod =1; FontSize=22;
YLabelFontSize =24;
FontSize= 24 ; LineWidth = 3;TitleFontSize = 20; LegendFontSize = 24; axis_ratio=1.5; 
myboldify(h,MarkerSize,YLabelFontSize,FontSize,LineWidth,LegendFontSize,TitleFontSize)