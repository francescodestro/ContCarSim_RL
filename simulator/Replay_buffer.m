classdef Replay_buffer
    
    properties
        storage
        max_size
    end

    methods

        function obj = Replay_buffer(max_size)
            obj.storage = {};
            obj.max_size = max_size;
        end

        function obj = push(obj, data)

            obj.storage = [obj.storage, data]; %data is a cell array = {{state, next_state, action, reward}}
            if length(data{1}) ~= 4
                warning('The size of the data is wrong');                
            end

            if length(obj.storage) > obj.max_size
                obj.storage(1:length(obj.storage)-obj.max_size) = [];
            end

        end

        function [x, y, u, r] = sample(obj, batch_size)
            x = []; y = []; u = []; r = []; %x:state, y:next_state, u:action, r:reward, 1-d:done
            if isempty(obj.storage)
                warning('The buffer is empty. Exiting the program.');
                return;
            end

            ind = randi([1 length(obj.storage)], 1 , batch_size);  
            for i = ind
                data = obj.storage{i};
                x(:,end+1) = data{1};
                y(:,end+1) = data{2};
                u(:,end+1) = data{3};
                r(:,end+1) = data{4};
            end
        end
    end

end