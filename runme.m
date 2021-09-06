clc
clear
addpath(genpath(pwd));

% DLSA deals with semi-supervised partial label learning approach via Dependence-maximized Label Set Assignment

%parameters
K=10;
dim_para = 0.999;
T = 50;
mu = 0.5;
alpha = 0.05;
gama = 1;
beta = 1;
Maxiter = 10;

% Load the file containing the necessary inputs for calling the DLSA function
% Data demo is Lost with partition percentage p set as 0.5 (50% examples are unlabeled)

DATAPATH=strcat('data/Lost_partial_0.5.mat');

Accuracy_DLSA = DLSA(DATAPATH,K,dim_para,T,mu,gama,beta,alpha,Maxiter);





