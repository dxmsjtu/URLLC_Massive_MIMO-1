%%
function  plot_snr_bler(SNR_Matrix,BLER_Matrix)
LineStyles='-bs -ks -bo -ko -bp -kp -kv -ro -cs -md -gv -md -bs -gv --rp --ko -m< -yd --bs -c> --gv --rp --co --m< --kd --y>';
 %LineStyles='-bs --bs -gd --gd -kv --kv -ro --ro -bo --bo -gv --gv -kd --kd -rs --rs -cs --cs -mv --mv -md -rp --bs --gv --rp --ko -m< -yd --bs -c> --gv --rp --co --m< --kd --y>';
Rows=size(BLER_Matrix,1);
LineStyles=parse(LineStyles);
MarkerSize =6;
LineWidth = 1;
for i=1:Rows
    %plot(SNR_Matrix,BLER_Matrix(i,:),LineStyles(i,:),'LineWidth',LineWidth,'MarkerSize',MarkerSize);
    semilogy(SNR_Matrix,BLER_Matrix(i,:),LineStyles(i,:),'LineWidth',LineWidth,'MarkerSize',MarkerSize);
    hold on;
    grid on;
end
axis([min(min(SNR_Matrix)) max(max(SNR_Matrix)) min(min(BLER_Matrix)) max(max(BLER_Matrix))  ]);
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