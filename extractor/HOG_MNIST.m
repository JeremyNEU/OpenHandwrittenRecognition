%%Author:Ethan Y H ZHANG
%%Date:2015-10-14
%%HOG features for MNIST

%%Loading files
function FeatureVector=HOG_MNIST(ima)
    patch_xnum=4;
    patch_ynum=4;
    SecondFeature=zeros(patch_xnum*patch_ynum,1);
    FeatureVector=zeros(80,1);
    image=zeros(30,30);
    image(2:29,2:29)=ima;
    for i=1:patch_xnum
        for j=1:patch_ynum
            feature=zeros(5,1);
            for k=1:7
                x=(i-1)*7+1+k;
                %disp(x);
                for p=1:7;
                    y=(j-1)*7+1+p;
                    %disp(y);
                    %caculate the gradient from horizonal and vertical
                    %direction;
                    Sx=(image(x-1,y-1)+2*image(x-1,y)+image(x-1,y+1))-(image(x+1,...
                        y-1)+2*image(x+1,y)+image(x+1,y+1));
                    Sy=(image(x-1,y-1)+2*image(x,y-1)+image(x+1,y-1))-(image(x-1,...
                        y+1)+2*image(x,y+1)+image(x+1,y+1));
                    if Sx==0 && Sy>0
                        theta=pi/2;
                    else if Sx == 0 && Sy < 0
                            theta=-pi/2;
                        else
                            theta=atan(Sy/Sx);
                        end
                    end
                    quantity=sqrt(Sx^2+Sy^2);
                    theta=theta+pi/2;
                    %disp(theta);
                    %======================================================
                    %assign the theta to the block of (0,pi)
                    if theta<=pi/5 
                        feature(1)=feature(1)+quantity;
                        else if theta<=2*pi/5
                            feature(2)=feature(2)+quantity;
                            else if theta<=3*pi/5
                                feature(3)=feature(3)+quantity;
                                else if theta<=4*pi/5
                                    feature(4)=feature(4)+quantity;
                                    else
                                        feature(5)=feature(5)+quantity;
                                    end
                                end
                            end
                    end  
                    %======================================================
                    SecondFeature(4*(i-1)+j)=SecondFeature(4*(i-1)+j)+image(x,y)/49;
               end
            end
            %add it to image feature;
            st=20*(i-1)+5*(j-1)+1;
            en=20*(i-1)+5*(j-1)+5;
            FeatureVector(st:en,1)=feature;    
        end
    end
%     FeatureVector(81:end,1)=SecondFeature;
end