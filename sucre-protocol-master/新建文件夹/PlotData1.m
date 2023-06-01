figure
%%
hold on;

plot(K0values,meanWaitingTimeACBPC(:,1),'r-.+','LineWidth',1);
plot(K0values,meanWaitingTimeACBPC(:,2),'b-.+','LineWidth',1);
plot(K0values,meanWaitingTimeBaseline,'k--','LineWidth',1);
xlabel('Number of Inactive UEs');
ylabel('Average Number of Access Attempts');
legend('ACBPC: No interf.','ACBPC: Interf.','Baseline','Location','SouthEast');
ylim([0 10]);
