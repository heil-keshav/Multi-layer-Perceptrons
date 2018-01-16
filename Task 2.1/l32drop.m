function l3b
close all;
clear all;
clc;
%directory containing dataset
folder = 'steering/';
filename = 'datanew.txt';

file_id = fopen(strcat(folder,filename));
Temp = textscan(file_id,'%s %f');
fclose(file_id);

% temp = size(Temp{1});
Y_full = Temp{2};
Y_full = Y_full';

files = dir(strcat(folder,'*.jpg'));
number_of_images = length(files);

% image dimensions
% Getting image dimensions from first image
% Assuming same image dimensions for complete dataset
temp = imread(strcat(folder,files(2).name));
s = size(temp);
height = s(1);
width = s(2);
    
% Initialise training data
X_full = zeros(width*height,number_of_images);
for i=1:number_of_images
    X_full(:,i) = reshape(rgb2gray(imread(strcat(folder,files(i).name)))',[1 1024]);
end
X_mean = mean(X_full);
X_std = std(X_full);
for i=1:(width*height)
    X_full(i,:) = X_full(i,:) - X_mean(1,:);
    X_full(i,:) = X_full(i,:)./X_std(1,:);
end
% Input Nodes
D=1024;

% Hidden layer 1
H1=512;

%Hidden layer 2
H2=64;

% number of epochs for training
nEpochs = 1000;

% learning rate
eta = 0.001;

figure;
plot(X_full(1024,:), 'b:', 'LineWidth', 2);
size(X_full)
% train the MLP using the generated sample dataset
[w1, w2, v, trainerror, validerror] = mlptrain(X_full, Y_full, D, H1, H2, eta, nEpochs);

figure;
hold on
plot(1:nEpochs, trainerror, 'b:', 'LineWidth', 2);
xlabel('Epochs');
ylabel('Error Value');
title('Sum of Squares Error Vs Training Iterations');

plot(1:nEpochs, validerror, 'r:', 'LineWidth', 2);
legend('Training error', 'Validation error');
hold off




function [w1, w2, v, trainerror, validerror] = mlptrain(X_full, Y_full, D, H1, H2, eta, nEpochs)
% X_full - training data of size Dx22000  D=1024
% Y_full - training labels of size 1x22000
% eta - the learning rate
% nEpochs - the number of training epochs

% define and initialize the neural network parameters

% weights for the connections between input and hidden layer
% random values from the interval [-0.01 0.01]
% w1 is a H1x(D+1) matrix
w1 = -0.01+(0.02)*rand(H1,D);
% Adding bias term =0
w1 = [zeros(H1,1) w1];

% weights for the connections between hidden layers
% random values from the interval [-0.01 0.01]
% w2 is a H2x(H1+1) matrix
w2 = -0.01+(0.02)*rand(H2,H1);
% Adding bias term =0
w2 = [zeros(H2,1) w2];

% weights for the connections between input and hidden layer
% random values from the interval [-0.01 0.01]
% v is a 1x(H2+1) matrix
v = -0.01+(0.02)*rand(1,H2);
% Adding bias term =0
v = [zeros(1,1) v];

% MiniBatch Size B
B=64;
[xa xb] = size(X_full);

input_taken = xb;
% Split set in 80-20 training/validation

tdata = floor(0.8*input_taken);
% vdata = rest of the input data;
X_train = X_full(:,1:tdata);
[m,n] = size(X_train);
X_valid = X_full(:,(tdata+1):input_taken);
Y_train = Y_full(1,1:tdata);
[y1, yt] = size(Y_train);
Y_valid = Y_full(1,(tdata+1):input_taken);
[y1, yv] = size(Y_valid);
for epoch=1:nEpochs
    B1 = 1;
    train_batches = ceil(n/B);
    for i=1:train_batches
        X = X_train(:,B1:min((B1+B-1),n));
        [a,b] =size(X);
        Y = Y_train(1,B1:min((B1+B-1),n));
        B1 = B1+B;
        
        % Dropout Probability
        p=0.5;
        
        drop = randperm(D, D*p);
        X(drop,:) = 0;
        % forward pass
        % --------------
        % Adding bias term to input
         X_1 = [ones(1, b); X];
        % input to hidden layer 1
        % calculate the output of the hidden layer one - z1
        % ---------
        %'TO DO'%
       
        z1 = w1*X_1;
        H_1 = sigmf(z1,[1,0]);
        
        % ---------
        drop = randperm(H1, H1*p);
        H_1(drop,:) = 0;
        
        % Add Bias in layer 1
        H_1 = [ones(1, b); H_1];
        
        % hidden1 to hidden2 layer
        % calculate the output of the hidden layer two - z2
        % ---------
        %'TO DO'%
       
        z2 = w2*H_1;
        H_2 = sigmf(z2,[1,0]);
       
        % ---------
        drop = randperm(H2, H2*p);
        H_2(drop,:) = 0;
        
        % Add Bias in layer 2
        H_2 = [ones(1, b); H_2];
        
        
        % hidden to output layer
        % calculate the output of the output layer units - ydash
        % ---------
        %'TO DO'%
       
        O = v*H_2;

            
        % ---------
  
        % backward pass
        % ---------------
        % update the weights for the connections between hidden and
        % outlayer units
        % ---------
        %'TO DO'%
        del3 = O-Y;
        v = v - (eta/B)*(del3*H_2');
       
        % ---------
        
        % update the weights for the connections between hidden 2 and
        % hidden 1 layer units
        % ---------
        %'TO DO'%
        del2 = (v'*del3).*H_2.*(1-H_2);    
        del2(1,:) = [];
        w2 = w2 -(eta/B)*(del2*(H_1'));
       
        % ---------
        
        % update the weights for the connections between the input and
        % hidden 1 layer units
        % ---------
        %'TO DO'%
        del1 = (w2'*del2).*H_1.*(1-H_1);
        del1(1,:) = [];
        w1 = w1 - (eta/B)*(del1*X_1');
        % ---------
    end
    % On training set
    ydash = mlptest(X_train, B, w1, w2, v);
    
    % compute the training error
    terror = 0.5*sum(((ydash-Y_train).*(ydash-Y_train)))/yt;
    trainerror(epoch) = terror;
     fprintf('training error after epoch %d: %f\n',epoch,terror);

    % On validation Set
    
    ydash = mlptest(X_valid, B, w1, w2, v);
    % compute the validation error
    verror = 0.5*sum(((ydash-Y_valid).*(ydash-Y_valid)))/yv;
    validerror(epoch) = verror;
    fprintf('validation error after epoch %d: %f\n',epoch,verror);
    fprintf('\n');
end

return;

function ydash = mlptest(X_data, B, w1, w2, v)
 B1 = 1;
 
 [m,n] = size(X_data);
    for i=1:ceil(n/B)
       
        X = X_data(:,B1:min((B1+B-1),n));
        [a,b] = size(X);
        % forward pass
        % --------------
        % Adding bias term to input
         X_1 = [ones(1, b); X];
        % input to hidden layer 1
        % calculate the output of the hidden layer one - z1
        % ---------
        %'TO DO'%
        z1 = w1*X_1;
        H_1 = sigmf(z1,[1,0]);
        % ---------
        % Add Bias in layer 1
        H_1 = [ones(1, b); H_1];
        
        % hidden1 to hidden2 layer
        % calculate the output of the hidden layer two - z2
        % ---------
        %'TO DO'%
        z2 = w2*H_1;
        H_2 = sigmf(z2,[1,0]);
        % ---------
        % Add Bias in layer 2
        H_2 = [ones(1, b); H_2];
        
        
        % hidden to output layer
        % calculate the output of the output layer units - ydash
        % ---------
        %'TO DO'%
        ydash(1,B1:min((B1+B-1),n)) = v*H_2;
        
        B1 = B1+B;
        % ---------
    end
return;
