
% ��ȡ��ȡ��������
[trLabel, trFea]=libsvmread('trFeature.txt');
[teLabel, teFea]=libsvmread('teFeature.txt');

%[bestCVaccuracy,bestc,bestg]=SVMcgForClass(trLabel,TrFea,cmin,cmax,gmin,gmax,v,cstep,gstep,accstep);
model=svmtrain(trLabel,trFea, '-c 1 -g 0.07');  %����-c -g�Ĳ���
[predict_label, accuracy, dec_values] =svmpredict(teLabel, teFea, model);