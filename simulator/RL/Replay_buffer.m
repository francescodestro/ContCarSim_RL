classdef Replay_buffer
    
    properties
        storage
        max_size
        ptr
    end

    methods
        function obj = Replay_buffer(max_size)
            obj.storage = {};
            obj.max_size = max_size;
            obj.ptr = 1;
        end

        function obj = push(obj, data)

            if length(obj.storage) == obj.max_size
                obj.storage{round(obj.ptr)} = data;
                obj.ptr = rem(obj.ptr, obj.max_size)+1;
            else
                obj.storage{end+1} = data;
            end

        end

        function [x, y, u, r, d] = sample(obj, batch_size)
            x = []; y = []; u = []; r = []; d = []; %x:state, y:next_state, u:action, r:reward, 1-d:done
            ind = randi([1 length(obj.storage)], 1 , batch_size);  
            for i = ind
                data = obj.storage{i};
                x(:,end+1) = data{1};
                y(:,end+1) = data{2};
                u(:,end+1) = data{3};
                r(:,end+1) = data{4};
                d(:,end+1) = data{5};
            end
        end
    end

end