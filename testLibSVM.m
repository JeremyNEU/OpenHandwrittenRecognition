
% 读取提取到的特征
[trLabel, trFea]=libsvmread('trFeature.txt');
[teLabel, teFea]=libsvmread('teFeature.txt');

%[bestCVaccuracy,bestc,bestg]=SVMcgForClass(trLabel,TrFea,cmin,cmax,gmin,gmax,v,cstep,gstep,accstep);
model=svmtrain(trLabel,trFea, '-c 1 -g 0.07');  %更改-c -g的参数
[predict_label, accuracy, dec_values] =svmpredict(teLabel, teFea, model);