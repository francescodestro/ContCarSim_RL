classdef DDPG

    properties
        state_dim 
        action_dim
        min_action
        max_action
        max_size %buffer size
        exploration_noise %exploration noise
        update_iteration %number of iterations for gradient descent
        batch_size
        gamma %discount factor for rewards
        tau %soft parameters for target networks
        lr_critic %learning rate for critic network
        lr_actor %learning rate for actor network

        actor
        actor_target

        critic
        critic_target

        replay_buffer

        num_critic_update_iteration
        num_actor_update_iteration
        num_training
        
        %agent's memory of the process
        state_pre %remember the last state
        action_pre %remember the last action
        reward_pre %remember the last reward
        done_pre %remember the last status of "done"
    end

    methods
        function obj = DDPG(state_dim, action_dim, min_action, max_action, ...
                max_size, exploration_noise, update_iteration,...
                batch_size, gamma, tau, lr_critic, lr_actor)
            
            obj.state_dim = state_dim;
            obj.action_dim = action_dim;
            obj.min_action = min_action;
            obj.max_action = max_action;
            obj.max_size = max_size;
            obj.exploration_noise = exploration_noise;
            obj.update_iteration = update_iteration;
            obj.batch_size = batch_size;
            obj.gamma = gamma;
            obj.tau = tau;
            obj.lr_critic = lr_critic;
            obj.lr_actor = lr_actor;

            obj.actor = Actor(state_dim, action_dim);
            obj.actor_target = Actor(state_dim, action_dim);

            obj.critic = Critic(state_dim, action_dim);
            obj.critic_target = Critic(state_dim, action_dim);

            obj.replay_buffer = Replay_buffer(max_size);

            obj.num_critic_update_iteration = 1;
            obj.num_actor_update_iteration = 1;
            obj.num_training = 1;

            obj.state_pre = zeros(state_dim,1);
            obj.action_pre = zeros(action_dim,1);
            obj.reward_pre = 0;
            obj.done_pre = 0;
        end

        function action = select_action(obj, state)
            dlstate = dlarray(state,'CB');
            action = forward(obj.actor.NN, dlstate);
            %scale [0,1] it to [min_action, max_action]
            action = action.*(obj.max_action-obj.min_action)+obj.min_action;
        end

        function obj = update(obj)
            averageGrad_critic = [];
            averageSqGrad_critic = [];
            averageGrad_actor = [];
            averageSqGrad_actor = [];
            for i=1:obj.update_iteration

                % Sample replay buffer
                [x, y, u, r, d] = obj.replay_buffer.sample(obj.batch_size);
                state = dlarray(x, 'CB');
                action = dlarray(u, 'CB');
                next_state = dlarray(y, 'CB');
                done = dlarray(1-d, 'CB');
                reward = dlarray(r, 'CB');
                
                if canUseGPU
                    state = gpuArray(state);
                    action = gpuArray(action);
                    next_state = gpuArray(next_state);
                    done = gpuArray(done);
                    reward = gpuArray(reward);
                end

                % Compute the target Q value
                next_action = forward(obj.actor_target.NN, next_state);
                next_action = next_action.*(obj.max_action-obj.min_action)+obj.min_action;
                target_Q = forward(obj.critic_target.NN, [next_state; next_action]);
                target_Q = reward + (done.*obj.gamma.*target_Q);
                
                % Compute critic loss and gradient 
                [loss_critic, gradients_critic] = dlfeval(@compute_critic_gradients, obj.critic.NN,...
                    state, action, target_Q);
                [obj.critic.NN,averageGrad_critic,averageSqGrad_critic] = adamupdate(obj.critic.NN, gradients_critic,...
                    averageGrad_critic,averageSqGrad_critic, i, obj.lr_critic, 0.75, 0.95);

                % Compute actor loss and gradient 
                [loss_actor, gradients_actor] = dlfeval(@compute_actor_gradients, obj.actor.NN, obj.critic.NN, state);
                [obj.actor.NN,averageGrad_actor,averageSqGrad_actor] = adamupdate(obj.actor.NN, gradients_actor,...
                    averageGrad_actor,averageSqGrad_actor, i, obj.lr_actor, 0.75, 0.95);

                % Soft update of target networks
                obj.actor_target.NN = softUpdate(obj.actor_target.NN, obj.actor.NN, obj.tau);
                obj.critic_target.NN = softUpdate(obj.critic_target.NN, obj.critic.NN, obj.tau);

                obj.num_actor_update_iteration = obj.num_actor_update_iteration + 1;
                obj.num_critic_update_iteration = obj.num_critic_update_iteration + 1;

            end

        end

        function save(obj)
            actor_weights = obj.actor.NN.Learnables;
            save('actor_weights.mat', 'actor_weights'); % Save trained network

            critic_weights = obj.critic.NN.Learnables;
            save('critic_weights.mat', 'critic_weights'); % Save trained network
        end

        function obj = load(obj)
            load('actor_weights.mat', 'actor_weights'); % Save trained network
            obj.actor.NN.Learnables = actor_weights;
        end

    end

end


function targetNet = softUpdate(targetNet, sourceNet, tau)
    for i = 1:numel(targetNet.Learnables.Value)
        targetNet.Learnables.Value{i} = (1 - tau)*targetNet.Learnables.Value{i} + tau*sourceNet.Learnables.Value{i};
    end
end

function [loss, gradients] = compute_critic_gradients(net, state, action, target_Q)
    
    current_Q = forward(net, cat(1, state, action)); % Get current Q estimate
    loss = mean((current_Q - target_Q).^2, 'all'); % Compute loss
    gradients = dlgradient(loss, net.Learnables); % Compute gradients

end


function [loss, gradients] = compute_actor_gradients(actorNN, criticNN, state)
    action = forward(actorNN, state);
    action = action.*(obj.max_action-obj.min_action)+obj.min_action;
    loss = -mean(forward(criticNN, cat(1, state, action))); % Compute actor loss
    gradients = dlgradient(loss, actorNN.Learnables); % Compute gradients

end