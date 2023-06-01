clear all; close all;
filenameTmp ='NeibourCellUser_10_60PL_exp_range';%可将仿真参数作为文件名
mkdir_str=strcat('.\Simulation\',filenameTmp);
mkdir(mkdir_str);%一运行就会在当前文件夹下创建simulation文件夹
mkdir_str1 =strcat(mkdir_str,'\');
mkdir_str =strcat(mkdir_str1,filenameTmp);
mkdir_str =strcat(mkdir_str,'.m');
strsave= strcat('.\Simulation\',filenameTmp,'\');
strsave= strcat(strsave,filenameTmp); s=['load ' strsave]; eval(s);% 保持.mat 文件，以后仿真结果可以再次确认,以后一定注意可以再次画图。
close all;
MarkerSize=9; LineWidth =2; LineMethod =1; PlotMethod =0; FontSize=22;
% DataMatrix(end+1,:) =meanWaitingTimeBaseline.';
SNR_Matrix =repmat(K0values,size(DataMatrix,1),1);
SubCaseNum = 16;%size(DataMatrix,1) ;    %size(DataMatrix,1);

NumofFiles =SubCaseNum;
for i=1:size(DataMatrix,1)/SubCaseNum;
    figure;
    DataMatrix_tmp =DataMatrix((i-1)*SubCaseNum+1:i*SubCaseNum,:);
    SNR_Matrix_tmp =SNR_Matrix((i-1)*SubCaseNum+1:i*SubCaseNum,:);
    selectedIndex =[1:2:SubCaseNum]
    DataMatrix_tmp =DataMatrix_tmp(selectedIndex,:);
    SNR_Matrix_tmp =SNR_Matrix_tmp(selectedIndex,:);
    NumofFiles =size(SNR_Matrix_tmp,1);
    Plot_SIC_results=Plot_SIC(MarkerSize,LineWidth,FontSize,SNR_Matrix_tmp,DataMatrix_tmp,NumofFiles,LineMethod,PlotMethod)'
    %%Plot simulation results
    legend('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20');
end
