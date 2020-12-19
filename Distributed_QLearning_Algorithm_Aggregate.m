% Distributed_QLearning_Algorithm_Aggregate
clear;
close all;
clc;
%% 
%Parameters
alpha = 0.5;
gamma = 0.9;
epsilon = 0.1;
n_iterations = 6000;
P = [-80,-50,-30,10,29.8]; % P: set of possible power levels (p), P = {p1...pl}
n_states = length(P);
SINR_th = 23;
num_SU = 10;

% Low/Medium/High distance
% est_attenuation = 20*log10(D) dB (subtract from signal power)
% D = [1 100 10000];


%Q Matrix
Q = zeros(n_states,n_states,num_SU);
Q_iterations = zeros(n_states,n_states,n_iterations+1);
for k = 1:num_SU
Q_iterations(:,:,1,k) = Q(:,:,k);
end

%Reward Matrix
for i = 1:length(P)
    SINR_vec(i) = SINR_X_UL(P(i));
end

cost_vec = cost(SINR_vec,SINR_th);
% cost_vec = cost_abs(SINR_vec,SINR_th); (|Difference|)
% cost_vec = cost_4(SINR_vec,SINR_th); (Difference^4)

R = zeros(n_states,n_states,num_SU);

%% Q Learning Algorithm 

state = randi(length(R(1,:,1)),1,num_SU);
action = zeros(1,num_SU);
for i = 1:n_iterations
    for k = 1:num_SU
        %Create R matrix holding other states constant
        p_mat = create_p_mat(state,k,P);
        for j = 1:length(P)
            SINR_vec(j) = SINR_X_UL(p_mat(j,:));
        end
        cost_vec = cost(SINR_vec,SINR_th);
        for j = 1:n_states
                R(j,:,k) = cost_vec;
        end
        
        %Random Greedy Algorithm choice for next Action
        random_greedy = rand;
        if random_greedy < epsilon
            action(k) = randi(length(R(1,:,1)));
        else
            min_Q = min(Q(state(k),:,k));
            possible = find(Q(state(k),:,k) == min_Q,n_states);
            action(k) =  possible(randi(length(possible)));
        end
        
        %Calculating Q and Stochastic Gradiant Descent 
        min_next_Q = min(Q(action(k),:,k));
        discount = gamma*min_next_Q;
        delta_Q = alpha*(R(state(k),action(k),k) + discount - Q(state(k),action(k),k));
        Q(state(k),action(k),k) = Q(state(k),action(k),k) + delta_Q;
        Q_iterations(:,:,i+1,k) = Q(:,:,k);
        state(k) = action(k);
    end
end

%% Analysis

%Normalize final Q table
for k = 1:num_SU
normal_factor = 100/max(Q(:,:,k),[],'all');
normalized_Q(:,:,k)  = round(normal_factor*Q(:,:,k));
end

%Plots 
Q_plot = squeeze(Q_iterations(3,1,:,1));
Q_plot_2 = squeeze(Q_iterations(1,2,:,4));
plot(Q_plot);
hold on
plot(Q_plot_2);
%Differences from example: greedy approach, any state can take any action.

% starting_state = randi(length(R(1,:)));
% end_Q = useQ(normalized_Q,starting_state);

%% Functions

function final_state = useQ(Q,starting_state)
    final_state = starting_state;
    for i = 1:10
        min_Q = min(Q(final_state,:));
        possible = find(Q(final_state,:) == min_Q,length(Q(final_state,:)));
        action =  possible(randi(length(possible)));
        if Q(final_state,action) == min(min(Q))
            final_state = action;
            break
        else
            final_state = action;
        end
    end
end

function p_mat = create_p_mat(current_states,omitted_state,P)
    for i = 1:length(current_states)
        p_vec(i) = P(current_states(i));
    end
    p_mat = zeros(length(P),length(current_states));
    for i = 1:length(P)
        p_mat(i,:) = p_vec;
    end
    p_mat(:,omitted_state) = P;
end

% SINR_X_UL
function SINR = SINR_X_UL(P_SU)
P_DTV = 90;
var = 0;

% future versions: gains will be vectors based on distance
h_DTV = 1;
h_SU = 1;  

SINR = (P_DTV*h_DTV) / (var + sum(P_SU.*h_SU));

end

% SINR_X_DL
function SINR = SINR_X_DL(P_BS)
P_DTV = 90;
var = 0;

% future versions: gains will be vectors based on distance
h_DTV = 1;
h_BS = 1;   

SINR = (P_DTV*h_DTV) / (var + sum(P_BS.*h_BS));
end

% SINR_BS_j
function SINR = SINR_BS_j(P_SU,j)
var = 0;

% future versions: gains will be vectors based on distance
h = 1; % this will be a row vector

SINR = (P_SU(j)*h) / (var + sum(P_SU.*h) - P_SU(j)*h);
% future verisons: use below
% SINR = (P_SU(j)*h(j)) / (var + sum(P_SU.*h) - P_SU(j)*h(j));
end

% SINR_SU_j
function SINR = SINR_SU_j(P_BS,j)
var = 0;

% future versions: gains will be vectors based on distance
h = 1; % this will be a row vector

SINR = (P_BS(j)*h) / (var + sum(P_BS.*h) - P_BS(j)*h);
% future verisons: use below
% SINR = (P_BS(j)*h(j)) / (var + sum(P_BS.*h) - P_BS(j)*h(j));
end

% Cost
function c = cost(SINR_ij,SINR_th)
    c = (SINR_ij - SINR_th).^2;
end

function c = cost_abs(SINR_ij,SINR_th)
    c = abs(SINR_ij - SINR_th);
end

function c = cost_4(SINR_ij,SINR_th)
    c = (SINR_ij - SINR_th).^4;
end