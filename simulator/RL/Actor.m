classdef Actor

    properties
        NN
    end

    methods
        function obj = Actor(state_dim, action_dim)
            Layers = [
                featureInputLayer(state_dim, 'Name', 'input') % Input layer (state_dim input features)
                fullyConnectedLayer(400, 'Name', 'fc1') % First fully connected layer
                reluLayer('Name', 'relu1') % Activation function
                fullyConnectedLayer(300, 'Name', 'fc2') % Second fully connected layer
                reluLayer('Name', 'relu2') % Activation function
                fullyConnectedLayer(action_dim, 'Name', 'output') % Output layer
                sigmoidLayer('Name','saturation')];   % output range is [0,1]
            
            obj.NN = dlnetwork(Layers);      
        end
    end
end
