clear all; close all;
global g_T_RA_REP g_SimTimes g_t1 g_RAOs g_TRAR g_WRAR g_NRAR g_WBO g_NPTmax g_TCR g_NHARQ g_THARQ g_TRQ g_K
global g_TA_RP g_TRP g_ProbRetransMsg g_ProbErrMsg

g_T_RA_REP = 5;
g_SimTimes = 10;  %仿真次数
g_t1 = 4;
g_RAOs = 54;  %可用preamble个数
g_K = 15;  %上行可授权PUSCH资源个数
g_TRAR = 2;
g_WRAR = 5;
g_NRAR = 3;
g_WBO = 21;  %回退窗大小
g_NPTmax = 10;  %preamble最大传输次数
g_TCR = 48;
g_NHARQ = 5;
g_THARQ = 4;
g_TRQ = 4;
g_TA_RP = 1;
g_TRP = 1;
g_ProbRetransMsg = 0.1;
g_TRAmax = 1 + (g_NPTmax - 1) * ceil((g_TRAR + g_WRAR + g_WBO) / g_T_RA_REP) * g_T_RA_REP;

deviceID = zeros();
preamble = zeros();
nthPreamble = zeros();
arrivalTime = zeros();
initArrivalTime = zeros();
delay = zeros();
preambleSuccess = zeros();
msgSuccess = zeros();
exclude = zeros();
RAR = zeros();

for index1 = 1:g_SimTimes
    simPc = 0;
    simPs = 0;
    simDa = 0;
    g_Tp = 10000;
   % M = [5000 10000 30000];
   M = [500 1000 3000]*0.1;
  
    msg4Fail=zeros(g_NHARQ);
    for index2 = 1:g_NHARQ
        msg4Fail(index2) = (g_ProbRetransMsg^(index2-1)) * (1-g_ProbRetransMsg)*(g_ProbRetransMsg^g_NHARQ);
    end
    msg4Fail=sum(msg4Fail);
    g_ProbErrMsg = g_ProbRetransMsg^g_NHARQ + msg4Fail;%msg错误传输概率
    
    cntRAR = 0;
    simRaSlot = g_Tp / g_T_RA_REP;
    
    for index3 = 1:length(M)
        simPc = 0;
        simPs = 0;
        simDa = 0;
        
        raSlot = 0;
        cntBackoff = 0;
        collidedRAOs = 0;
        successMS = 0;
        
        for index4 = 1:M(index3)
            deviceID(index4) = index4;
            preamble(index4) = randi([1,g_RAOs],1,1);
            nthPreamble(index4) = 1;
            arrivalTime(index4) = round(betarnd(3,4) * simRaSlot);  %通过改变到达时间的模型，可分别匹配3GPP给出的traffic model 1 和traffic model 2，即：traffic model 1对应rand(1,simRaSlot),traffic model 2对应round(betarnd(3,4) * simRaSlot)
            initArrivalTime(index4) = arrivalTime(index4);
            delay(index4) = 0;
            preambleSuccess(index4) = false;
            msgSuccess(index4) = false;
            exclude(index4) = false;
        end
        
        simTp = max(initArrivalTime);
        if mod((simTp-g_t1),5) == 0
            simTE = 0;
        else
            simTE = g_T_RA_REP - mod((simTp-g_t1),5);
        end
        
        offsetTp = g_t1 + (simTp+simTE)*5 + g_TRAmax;
        simIR = (offsetTp-g_t1)/g_T_RA_REP;
        simTRAI = simTp + simTE + g_TRAmax;
        
        for i = 0:(offsetTp - 1)
            if mod((i + 1),5)==0
                for j = 1:M(index3)
                    if arrivalTime(j) == raSlot && ~(exclude(j)) && ~(msgSuccess(j))
                        n = nthPreamble(j);
                        probDetecPreamble = 1 - 1 / exp(n);
                        if rand <= probDetecPreamble
                            for k = 1:M(index3)
                                if k ~= j && arrivalTime(k) == raSlot && ~(exclude(k)) && ~(msgSuccess(k))
                                    if preamble(j) == preamble(k)
                                        randBackoff =  randi([1,g_WBO],1,1);
                                        arrivalTime(k) = ceil((i + randBackoff + g_TRAR - g_t1) / 5);
                                        
                                        if arrivalTime(k) <= raSlot
                                            arrivalTime(k) = arrivalTime(k) + 1;
                                        end
                                        nthPreamble(k) = nthPreamble(k) + 1;
                                        
                                        if nthPreamble(k) > g_NPTmax
                                            exclude(k) = true;
                                        end
                                        preambleSuccess(k) = false;
                                        preamble(k) = randi([1,g_RAOs],1,1);
                                        
                                        cntBackoff = cntBackoff+1;
                                    end
                                end
                            end
                            
                            if cntBackoff == 0
                                RAR(cntRAR + 1) = deviceID(j);
                                cntRAR = cntRAR +1;
                            else
                                collidedRAOs = collidedRAOs +1;
                                randBackoff =  randi([1,g_WBO],1,1);
                                arrivalTime(j) = ceil((i + randBackoff - g_t1 + g_TRAR) / 5);
                                if arrivalTime(j) <= raSlot
                                    arrivalTime(j) = arrivalTime(j) + 1;
                                end
                                nthPreamble(j) = nthPreamble(j) + 1;
                                if nthPreamble(j) > g_NPTmax
                                    exclude(j) = true;
                                end
                                preambleSuccess(j) = false;
                                preamble(j) = randi([1,g_RAOs],1,1);
                                
                                cntBackoff = 0;
                            end
                        else
                            randBackoff =  randi([1,g_WBO],1,1);
                            arrivalTime(j) = ceil((i + randBackoff - g_t1 + g_TRAR) / 5);
                            if arrivalTime(j) <= raSlot
                                arrivalTime(j) = arrivalTime(j) + 1;
                            end
                            nthPreamble(j) = nthPreamble(j) + 1;
                            if nthPreamble(j) > g_NPTmax
                                exclude(j) = true;
                            end
                            preamble(j) = randi([1,g_RAOs],1,1);
                        end
                    end
                end
                
                if cntRAR > 0
                    repmat(RAR,[1 cntRAR]);
                elseif cntRAR == 0
                    RAR=[];
                end
                
                while (length(RAR) > g_K)
                    position = randi([1,length(RAR)],1,1);
                    randBackoff = randi([1,g_WBO],1,1);
                    arrivalTime(RAR(position)) = ceil((i + randBackoff - g_t1 + g_TRAR + g_WRAR) / 5);
                    
                    if nthPreamble(RAR(position)) <= g_NPTmax
                        nthPreamble(RAR(position)) = nthPreamble(RAR(position)) + 1;
                    end
                    
                    if nthPreamble(RAR(position)) > g_NPTmax
                        exclude(RAR(position)) = true;
                    end
                    
                    preamble(RAR(position)) = randi([1,g_RAOs],1,1);
                    
                    RAR(position) = [];
                end
                
                for index4 = 1:length(RAR)
                    preambleSuccess(RAR(index4)) = true;
                end
                cntRAR = 0;
                raSlot = raSlot + 1;
            end
            
            if mod((i - 2),5)==0 && i>10
                for index5 = 1:M(index3)
                    if preambleSuccess(index5) && ~(exclude(index5)) && ~(msgSuccess(index5))
                        if rand > g_ProbErrMsg
                            msgSuccess(index5) = true;
                            delay(index5) = i + 21 - (g_t1 + initArrivalTime(index5) * 5);
                        else
                            randBackoff = rand * g_WBO;
                            arrivalTime(index5) = ceil((i + 48 + randBackoff - g_t1) / 5);
                            
                            if arrivalTime(index5) <= raSlot
                                arrivalTime(index5) = raSlot;
                            end
                            
                            nthPreamble(index5) = nthPreamble(index5) + 1;
                            if nthPreamble(index5) > g_NPTmax
                                exclude(index5) = true;
                            end
                            preamble(index5) = randi([1,g_RAOs],1,1);
                            msgSuccess(index5) = false;
                            preambleSuccess(index5) = false;
                        end
                    end
                end
            end
        end
        
        for index6 = 1:M(index3)
            if msgSuccess(index6)
                successMS = successMS+1;
                simDa = simDa + delay(index6);
            end
        end
        
        cntNthPramble = 0;
        for index7 = 1:g_NPTmax
            for index8 = 1: M(index3)
                if  msgSuccess(index8) && nthPreamble(index8) <= index7
                    cntNthPramble = cntNthPramble+1;
                end
            end
        end
        
        simPc = collidedRAOs / (simIR * g_RAOs);
        simPs = successMS / M(index3);
        simDa = simDa / successMS;
        
        stoPc(index1,index3) = simPc;
        stoPs(index1,index3) = simPs;
        stoDa(index1,index3) = simDa;
        
    end
end
Pc = mean(stoPc)
Ps = mean(stoPs)
Da = mean(stoDa)