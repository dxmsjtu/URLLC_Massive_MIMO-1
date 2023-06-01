clear all;
close all;
MarkerSize=9; LineWidth =2; LineMethod =1; PlotMethod =1; FontSize=22;
% BLER =rand(6,5);
BLER= [  0.8635    0.7512    0.6013    0.5053    0.4225
    0.2920    0.4194    0.3212    0.6276    0.4677
    0.1335    0.0002    0.2843    0.7193    0.0226
    0.6727    0.1495    0.4353    0.0239    0.0651
    0.2026    0.2738    0.9038    0.5749    0.9240
    0.8685    0.8724    0.9251    0.0465    0.5341]
SNR_Matrix =repmat([1:5],size(BLER,1),1);
NumofFiles =size(BLER,1);
NumofFiles =size(BLER,1);
Plot_SIC_results=Plot_SIC(MarkerSize,LineWidth,FontSize,SNR_Matrix,BLER,NumofFiles,LineMethod,PlotMethod)
legend('Out-it =1, In-it =1','Out-it =1, In-it =3','Out-it =2, In-it =1','Out-it =2, In-it =3','Out-it =1');
h  =gcf; % 获得当前figure 句柄

FontSize= 24 ; LineWidth = 3;TitleFontSize =20; LegendFontSize =24; axis_ratio=1.5; %myboldify(h,FontSize,LineWidth,LegendFontSize,TitleFontSize);
myboldify(h,MarkerSize,YLabelFontSize,FontSize,LineWidth,LegendFontSize,TitleFontSize)

figure
Plot_SIC_results=Plot_SIC(MarkerSize,LineWidth,FontSize,SNR_Matrix,BLER,NumofFiles,LineMethod,PlotMethod)
legend('Out-it =1, In-it =1','Out-it =1, In-it =3','Out-it =2, In-it =1','Out-it =2, In-it =3','Out-it =1');

h  =gcf; % 获得当前figure 句柄，大家需要用这个模板来画图，仔细调整写出的文章才
MarkerSize=9; LineWidth =2; LineMethod =1; PlotMethod =1; FontSize=22;
YLabelFontSize =24;
FontSize= 24 ; LineWidth = 3;TitleFontSize = 20; LegendFontSize = 24; axis_ratio=1.5; %myboldify(h,FontSize,LineWidth,LegendFontSize,TitleFontSize);
myboldify(h,MarkerSize,YLabelFontSize,FontSize,LineWidth,LegendFontSize,TitleFontSize)

