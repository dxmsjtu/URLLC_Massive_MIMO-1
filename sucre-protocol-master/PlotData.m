figure
%%
subplot(121)
hold on; box on;
plot(K0values,meanWaitingTimeACBPC(:,1),'r-.+','LineWidth',1);
plot(K0values,meanWaitingTimeACBPC(:,2),'b-.+','LineWidth',1);
plot(K0values,meanWaitingTimeBaseline,'k--','LineWidth',1);
xlabel('Number of Inactive UEs');
ylabel('Average Number of Access Attempts');
legend('ACBPC: No interf.','ACBPC: Interf.','Baseline','Location','SouthEast');
ylim([0 10]);

subplot(122)
hold on; box on;
yyaxis left;
plot(K0values,finalWaitingTimesACBPC(end,:,1)./sum(finalWaitingTimesACBPC(:,:,1),1),'r-.+','LineWidth',1);
plot(K0values,finalWaitingTimesACBPC(end,:,2)./sum(finalWaitingTimesACBPC(:,:,2),1),'b-.+','LineWidth',1);
plot(K0values,finalWaitingTimesBaseline(end,:)./sum(finalWaitingTimesBaseline,1),'k--','LineWidth',1);
xlabel('Number of Inactive UEs');
ylabel('Fraction of Failed Access Attempts');
%legend('ACBPC: No interf.','ACBPC: Interf.','Baseline','Location','SouthEast');

yyaxis right;
plot(K0values,meanCollidingUsersNumACBPC(:,1),'r--x','LineWidth',1);
plot(K0values,meanCollidingUsersNumACBPC(:,2),'b--x','LineWidth',1);
xlabel('Number of Inactive UEs');
ylabel('Average Number of Colliding UEs');
legend('ACBPC: No interf.','ACBPC: Interf.','Baseline','ACBPC: No interf.','ACBPC: Interf.','Location','SouthEast');
ylim([0 40])