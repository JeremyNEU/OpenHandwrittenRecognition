%Nueral Network trainning and prediction quick one and correct one
function exNN()
    clear;clc;
    load('NNdata.mat');
    %load('ex4weights.mat');
    %initialize Theta1 and Theta2;
    %input layer units=400;output layer units=10;
    thr1=sqrt(6)/sqrt(400+25);
    thr2=sqrt(6)/sqrt(25+400);
    theta1=rand(120,1+400)*2*thr1-thr1;
    theta2=rand(400,1+120)*2*thr2-thr2;
    %basic settings
    y=X;
    g=inline('1./(1+exp(-z))');
    m=size(X,1);
    X=[ones(m,1),X];
    %-------------------implementation-----------------------
    %use iteration method to train theta;
    max_itr=250000;
    alpha=0.0001;
    alpha2=0.7;
    flag=1;
    flag1=1;
    y=g(y);
    for u=1:max_itr
        disp(u);
        %for each sample
        errh=0;
        erro=0;
            %hidden layer
            u_hidden_layer=X*theta1';
            u_hidden_layer=[ones(m,1) u_hidden_layer];
            y_hidden_layer=g(u_hidden_layer);
            %output layer
            u_output_layer=y_hidden_layer*theta2';
            y_output_layer=g(u_output_layer);
            %output layer Error
            Err_output=y_output_layer.*(1-y_output_layer).*(y-y_output_layer)...
                +alpha2*erro;
            %Error of each hidden layer
            Err_hidden=y_hidden_layer(:,2:end).*(1-y_hidden_layer(:,2:end)).*(Err_output*theta2(:,2:end))...
                +alpha2*errh;
            %update Wik
            theta2=theta2+alpha*Err_output'*y_hidden_layer;
            %update Wkj
            theta1=theta1+alpha*Err_hidden'*X;
            %last time's Err
            errh=Err_hidden;
            erro=Err_output;
%         if mod(u,10)==0
%             %========================================================
%             %========================================================
%             %see change every loop
%             %hidden layer
%             u_hidden_layer=X*theta1';
%             u_hidden_layer=[ones(m,1) u_hidden_layer];
%             y_hidden_layer=g(u_hidden_layer);
%             %output layer
%             u_output_layer=y_hidden_layer*theta2';
%             y_output_layer=g(u_output_layer);
%             index=ones(m,1);
%             for i=1:m
%                 temp_big=y_output_layer(i,1);
%                 for j=2:10
%                     if y_output_layer(i,j)>temp_big
%                         temp_big=y_output_layer(i,j);
%                         index(i)=j;
%                         if j==10
%                             index(i)=0;
%                         end
%                     end
%                 end
%             end
%             yy(1:500,1)=zeros(500,1);
%             cmp=[index,yy,index-yy];
%             for i=1:m
%                 if cmp(i,3)~=0
%                     cmp(i,3)=1;
%                 end
%             end
%             %cmp
%             disp(strcat('Accuracy Rate of times:',num2str(u)));
%             disp(strcat('--------------',num2str((1-sum(cmp(:,3))/m)*100),'%'));
%             disp(1/2*sum(sum((y-y_output_layer).^2)));
%             %========================================================
%             %========================================================
%             if 1-sum(cmp(:,3))/m>0.95
%                 save theta1;
%                 save theta2;
%             end            
%             if 1-sum(cmp(:,3))/m>0.8 
%                 if flag==1
%                     alpha=0.9*alpha;
%                     flag=0;
%                 end
%             end
%             if 1-sum(cmp(:,3))/m>0.9 
%                 if flag1==1
%                     alpha=0.9*alpha;
%                     flag1=0;
%                 end
%             end
%         end
    end
    save weight120 theta1;  
end