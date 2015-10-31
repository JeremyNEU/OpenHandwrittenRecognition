%% Author:Ethan Y H Zhang
% Date:2015-10-22
% Feature from Sparse self encoding NN

function FeatureVector=NNFeature(image)
    image=image(5:24,5:24);
    imaV=zeros(1,400);
    for i=1:20
        for j=1:20
            imaV(20*(i-1)+j)=image(i,j);
        end
    end
    imaV=[1 imaV];
% load weight trained by exNN.m
    load weight120.mat;
    FeatureVector=theta1*imaV';
end