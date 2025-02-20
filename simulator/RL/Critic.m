classdef Critic
    
    properties
        NN
    end

    methods
        function obj = Critic(state_dim, action_dim)
            Layers = [
                featureInputLayer(state_dim+action_dim, 'Name', 'input') % Input layer (10 input features)
                fullyConnectedLayer(400, 'Name', 'fc1') % First fully connected layer
                reluLayer('Name', 'relu1') % Activation function
                fullyConnectedLayer(300, 'Name', 'fc2') % Second fully connected layer
                reluLayer('Name', 'relu2') % Activation function
                fullyConnectedLayer(1, 'Name', 'output')]; % Output layer

            obj.NN = dlnetwork(Layers);
        end       
    end

end