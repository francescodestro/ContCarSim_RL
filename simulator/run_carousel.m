%% Script for running the carousel simulator
% Francesco Destro, October2021

clc, clear, %close all
rng default

%% set simulation conditions    
control_mode= 1; % variable passed to controller.m function, useful to test multiple control strategies
                    % implemented control strategies:
                    % 0: open-loop
                    % 1: RL: training
                    % 2: RL: deployment
disturbance_scenario = 1;  % 0: normal operating conditions; 1: nominal slurry concentration ramp; 2: cake resistance step
total_duration = 3600*2; % s

%% set nominal operating variables   
u_nominal.t_cycle=100;           % nominal set-point of cycle duration (s) 
                              % u_nominal.t_cycle and u.t_cycle MUST ALWAYS BE INTEGERS
u_nominal.V_slurry=3e-6;     % nominal set-point of fed slurry volume (m3) 
                                  % Note: can't be larger than 10e-6; to process
                                  % larger slurry volumes, comment lines 15-17
                                  % of run_simulation.m
u_nominal.P_compr=10e4;            % nominal set-point of gauge pressure compressor (Pa)
u_nominal.Tinlet_drying=50+273.15;      % nominal set-point of drying gas temperature (K) 

%% set nominal feed properties
cryst_output.conc_slurry=250;   % nominal slurry concentration (kg/m3) - 
                                % actual slurry concentration subject to Gaussian disturbances (+ ramp if disturbance_scenario==1)

%% set sampling interval and control time
control_interval = 1; % time step at which controller_online.m is called (s)
                        % MUST BE MULTIPLE Of 1 s                         
sampling_interval = .1; % sampling time for output measurements and states
                    % MUST BE SUBMULTIPLE OF 1 s

%% Set inter-cycle idle time and mesh cleaning idle time                    
inter_cycle_Dt = 0; % dead time at the end of every cycle (s); default = 0
mesh_clean_Dt  = 0; % dead time at mesh cleaning (s); default = 0

%% Create RL agent
% hyperparameters for DDPG
state_dim = 10;
action_dim = 2;

min_action = [5; 10]; %action(1)*1e-7 = V_slurry, action(2) = t_cycle
max_action = [100; 20*60];

max_size = 1e5; %buffer size
update_iteration = 200;
batch_size = 100;
gamma = 0.99; %discount factor
tau = 0.005;
lr_critic = 1e-2;
lr_actor = 1e-3;
exploration_noise = 0.2;

agent = DDPG(state_dim, action_dim, min_action, max_action, max_size, ...
    exploration_noise, update_iteration,... 
    batch_size, gamma, tau,...
    lr_critic, lr_actor);

% agent = load('agent.mat', 'agent').agent; % continue train the previous agent
% agent.max_size = max_size;
% agent.replay_buffer.max_size = max_size;
                    
%% run simulator
max_episode = 1e4;
interval_test = 10;

if control_mode==1
    for i=1:max_episode
        %run the carousel in parallel
        data = {}; %collect the data
        parfor j=1:maxNumCompThreads 
            data_buffer = {{zeros(state_dim),zeros(state_dim),zeros(action_dim),0.0}};
            simulation_output = run_simulation(u_nominal,cryst_output,disturbance_scenario,...
                control_mode,total_duration,control_interval,sampling_interval,...
                inter_cycle_Dt,mesh_clean_Dt,data_buffer,agent);
            data = [data, {simulation_output.data_buffer{2:end-1}}]; 
        end
        
        agent.replay_buffer = agent.replay_buffer.push(data);%push the data to agent's storage
        fprintf('Episode: \t%d \n', i);
    
        % start training
        agent = agent.update();
        %save and test the model
        if mod(i,interval_test)==0    
            test_mode = 2;
            data_buffer = {{zeros(state_dim),zeros(state_dim),zeros(action_dim),0.0}};

            simulation_output = run_simulation(u_nominal,cryst_output,disturbance_scenario,...
                test_mode,total_duration,control_interval,sampling_interval,...
                inter_cycle_Dt,mesh_clean_Dt, data_buffer, agent);

            total_reward = sum(cellfun(@(x) x{4}, simulation_output.data_buffer));
            fprintf('Episode: \t%d Total Reward of Test: \t%.2f\n', i, total_reward);
            agent.save()
            
        end
 
    end

elseif control_mode==0 %test the nominal control

    data_buffer = {{zeros(state_dim),zeros(state_dim),zeros(action_dim),0.0}};
    simulation_output = run_simulation(u_nominal,cryst_output,disturbance_scenario,...
        control_mode,total_duration,control_interval,sampling_interval,...
        inter_cycle_Dt,mesh_clean_Dt,data_buffer,agent);

elseif control_mode == 2 %test the learned control policy

    agent.load()
    data_buffer = {{zeros(state_dim),zeros(state_dim),zeros(action_dim),0.0}};
    simulation_output = run_simulation(u_nominal,cryst_output,disturbance_scenario,...
        control_mode,total_duration,control_interval,sampling_interval,...
        inter_cycle_Dt,mesh_clean_Dt,data_buffer,agent);

end
total_reward = sum(cellfun(@(x) x{4}, simulation_output.data_buffer));
fprintf('Episode: \t%d Total Reward of Test: \t%.2f\n', i, total_reward);


