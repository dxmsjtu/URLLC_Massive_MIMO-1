clc;clear all;close all;
a=load('RCUs_SP_ULA_Sweep_Angle_pd pilots_complete collide_SpatialCorrelation_LS_MR_UEs_2_SNR_0_np_2_n_300_M_100.mat');
b=load('RCUs_SP_ULA_Sweep_Angle_pd pilots_complete collide_SpatialCorrelation_MMSE_MR_UEs_2_SNR_0_np_2_n_300_M_100.mat');
c=load('RCUs_SP_ULA_Sweep_Angle_pd pilots_component collide_SpatialCorrelation_LS_MR_UEs_2_SNR_0_np_2_n_300_M_100.mat');
d=load('RCUs_SP_ULA_Sweep_Angle_pd pilots_component collide_SpatialCorrelation_MMSE_MR_UEs_2_SNR_0_np_2_n_300_M_100.mat');
e=load('RCUs_SP_ULA_Sweep_Angle_pd pilots_no collide_SpatialCorrelation_LS_MR_UEs_2_SNR_0_np_2_n_300_M_100.mat');
f=load('RCUs_SP_ULA_Sweep_Angle_pd pilots_no collide_SpatialCorrelation_MMSE_MR_UEs_2_SNR_0_np_2_n_300_M_100.mat');
DataMatrix_ul = [a.data.avg_error_ul b.data.avg_error_ul c.data.avg_error_ul d.data.avg_error_ul e.data.avg_error_ul f.data.avg_error_ul]';
%DataMatrix_ul = [a.data.avg_error_ul b.data.avg_error_ul c.data.avg_error_ul d.data.avg_error_ul]';
%DataMatrix_dl = [a.data.avg_error_dl b.data.avg_error_dl c.data.avg_error_dl d.data.avg_error_dl]';
DataMatrix_dl = [a.data.avg_error_dl b.data.avg_error_dl c.data.avg_error_dl d.data.avg_error_dl e.data.avg_error_dl f.data.avg_error_dl]';
figure(1); hold on;
plot_snr_bler(rad2deg(a.data.angle2),DataMatrix_ul)
title('UL')
xlabel('angle2', 'Interpreter','latex'); ylabel('Error probability', 'Interpreter','latex');
set(gca, 'YScale', 'log');set(gca,'Fontname','Monospaced'); set(gca,'FontSize',12);set(gca,'XTick',rad2deg(a.data.angle2));set(gca, 'XMinorGrid','on');
%legend('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30');
legend('complete collide MF','complete collide MMSE','component collide MF','component collide MMSE','no collide MF', 'no collide MMSE');
%legend('complete collide MF','complete collide MMSE','component collide MF','component collide MMSE');
h  =gcf; % 获得当前figure 句柄，大家需要用这个模板来画图，仔细调整写出的文章才
MarkerSize=9; LineWidth =2; LineMethod =1; PlotMethod =1; FontSize=22;
YLabelFontSize =24;
FontSize= 24 ; LineWidth = 3;TitleFontSize = 20; LegendFontSize = 24; axis_ratio=1.5; 
myboldify(h,MarkerSize,YLabelFontSize,FontSize,LineWidth,LegendFontSize,TitleFontSize)

figure(2); hold on;
plot_snr_bler(rad2deg(a.data.angle2),DataMatrix_dl)
title('DL')
xlabel('angle2', 'Interpreter','latex'); ylabel('Error probability', 'Interpreter','latex');
set(gca, 'YScale', 'log');set(gca,'Fontname','Monospaced'); set(gca,'FontSize',12);set(gca,'XTick',rad2deg(a.data.angle2));set(gca, 'XMinorGrid','on');
%legend('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30');
legend('complete collide MF','complete collide MMSE','component collide MF','component collide MMSE','no collide MF', 'no collide MMSE');
%legend('complete collide MF','complete collide MMSE','component collide MF','component collide MMSE');
h  =gcf; % 获得当前figure 句柄，大家需要用这个模板来画图，仔细调整写出的文章才
MarkerSize=9; LineWidth =2; LineMethod =1; PlotMethod =1; FontSize=22;
YLabelFontSize =24;
FontSize= 24 ; LineWidth = 3;TitleFontSize = 20; LegendFontSize = 24; axis_ratio=1.5; 
myboldify(h,MarkerSize,YLabelFontSize,FontSize,LineWidth,LegendFontSize,TitleFontSize)