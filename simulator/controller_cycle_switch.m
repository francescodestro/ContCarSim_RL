function [u,u_nominal,operating_vars, data_buffer] = controller_cycle_switch(process_time,cycle_time,...
               stations_working,u,u_nominal,cryst_output_nominal,measurements,operating_vars,x_estim,...
               n_cycle,control_mode, res_solvent, data_buffer, agent)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Inputs
    %
    % process_time      =   timer started at process onset (s)
    % cycle_time        =   timer re-started at every carousel rotation (s)
    % stations_working     =   vector [1x4] - for i=1:4:
    %                       - stations_working(i)=1 if station i was processing material during cycle that just finished ;
    %                       - stations_working(i)=0 if station i was empty during cycle that just finished; 
    % u                 =   vector of set-points of operating variables during previous control interval
    %                       Fields of u:
    %                       - u.t_cycle=cycle duration set-point (s)    MUST BE AN INTEGER
    %                       - u.V_slurry=fed slurry volume set-point (m3)
    %                       - u.P_compr= gauge pressure provided by compressor P101 (Pa)
    %                       - u.Tinlet_drying=drying gas temperature Station 5 set-point (K)                   
    % u_nominal         =   nominal value of manipulated variables, as set in run_carousel.m
    %                       or updated later. Same fields of u
    % cryst_output_nominal  = object containing nominal feed
    %                        conditions. Fields:
    %                        - cryst_output_nominal.conc_slurry= nominal
    %                                 slurry concentration in feed (kg/m3)
    %                        - cryst_output_nominal.x = Crystal size
    %                                 distribution – particles diameters (m)
    %                        - cryst_output_nominal.CSD - Volumetric crystal size distribution – percentage
    %                        - cryst_output_nominal.CSD_perc - Volumetric crystal size distribution – percentage
    %                        - cryst_output_nominal.T - slurry temperature (= room temperature) (K)  
    % 
    % measurements      =   object of process measurements since process onset 
    %                       with the sampling interval that has been set in run_carousel.m
    %                       Fields of measurements:
    %                       - measurements.t_meas = vector of sampling intervals (s)
    %                       - measurements.m_filt_WI101 = vector of filtrate mass measured by WI101 (kg)
    %                       - measurements.P_PI101 = vector of pressure measured by PI101 - gauge (Pa)
    %                       - measurements.P_PI102 = vector of pressure measured by PI102 - gauge (Pa)
    %                       - measurements.c_slurry_AI101 = vector of slurry concentration measured by AI101 (kg/m3)
    %                       - measurements.L_cake_LI101 = vector of height of cakes in Station 1 measured by LI101 (m)
    %                       - measurements.V_slurry_LI101 = vector of slurry volume in Station 1 measured by LI101 (m)
    %                       - measurements.Tg_in_TI101 = vector of temperatures of drying gas measured by TI101 (K) - inlet
    %                       - measurements.Tg_out_TI102 = vector of temperatures of drying gas measured by TI102 (K) - outlet                      
    %                       - measurements.Vdryer_FI101 = vector of drying gas flowrate measured by FI101 (m3/s)
    % operating_vars  =   object storing the profiles of the manipulated variables (automatically updated)
    %                       Fields of operating_vars:
    %                       - operating_vars.t_vector = control times vector
    %                       - operating_vars.P_compr_vector = u.P_compr time profile [1 x length(operating_vars.t_vector)]
    %                       - operating_vars.Tin_drying_vector = u.Tinlet_drying time profile [1 x length(operating_vars.t_vector)]
    %                       - operating_vars.n_cycle_vector = list of number of initialized carousel cycles
    %                       - operating_vars.t_cycle_vector = u.t_cycle time profile [1 x length(operating_vars.n_cycle_vector)]
    %                       - operating_vars.V_slurry_vector = u.V_slurry time profile [1 x length(operating_vars.n_cycle_vector)]
    % x_estim           =   object containing states and parameters estimated by estimator_online.m and estimator_cycle_switch
    % n_cycle           =   cycle counter - number of cycle that has just finished
    % control_mode      =   scalar defined in run_carousel.m
    %
    % Outputs           
    % u                 =   vector of manipulated variables for following control interval
    %                       Fields of u:
    %                       - u.t_cycle=cycle duration set-point (s)    MUST BE AN INTEGER
    %                       - u.V_slurry=fed slurry volume set-point (m3)
    %                       - u.P_compr= gauge pressure provided by compressor P101 (Pa)
    %                       - u.Tinlet_drying=drying gas temperature Station 4 set-point (K)   
    %         -------->     Fields not updated during call to this function
    %                       will retain the value set for the previous control interval
    % u_nominal         =   nominal value of manipulated variables, as set in run_carousel.m
    %                       or updated later. Same fields of u
    % operating_vars  =   object storing the profiles of the manipulated variables (automatically updated)
    %                       Fields of operating_vars:
    %                       - operating_vars.t_vector = control times vector
    %                       - operating_vars.P_compr_vector = u.P_compr time profile [1 x length(operating_vars.t_vector)]
    %                       - operating_vars.Tin_drying_vector = u.Tinlet_drying time profile [1 x length(operating_vars.t_vector)]
    %                       - operating_vars.n_cycle_vector = list of number of initialized carousel cycles
    %                       - operating_vars.t_cycle_vector = u.t_cycle time profile [1 x length(operating_vars.n_cycle_vector)]
    %                       - operating_vars.V_slurry_vector = u.V_slurry time profile [1 x length(operating_vars.n_cycle_vector)]
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%         
    
    %% call reinforcement learning algorithm
    % remember that u.t_cycle must always be an integer

    % RL inputs:
    %   - u.Tinlet_drying: inlet drying gas temperature (scalar)
    %   - u. P_compr: inlet drying gas pressure (scalar) 
    %   - measurements.c_slurry_AI101: inlet slurry concentration (time profile over the previous cycle, or from the beginning of the process?)
    %   - measurements.m_filt_WI101: filtrate mass (time profile over the previous cycle, or from the beginning of the process?)
    %   - measurements.Tg_out_TI102: outlet drying gas temperature (time profile over the previous cycle, or from the beginning of the process?)
    %   - stations_working: 0 (inactive) or 1 (active) if the station is working in the next cycle [1x4 array]
    %   - res_solvent: residual solvent content in discharged cakes (scalar or vector?). Note that cakes are not discharged at every cycle

    % measurements (RL state)
    
    state = [];
    
    state(end+1) = u.Tinlet_drying/323.0-1.0;
    state(end+1) = u.P_compr/1e5-1.0;
    state(end+1) = mean(measurements.c_slurry_AI101)/252-1.0;

    %sample from the profile
    sample_num = 1;
    if length(measurements.m_filt_WI101) > 1
        indices = round(linspace(1, length(measurements.m_filt_WI101), sample_num));
        state(end+1:end+sample_num) = measurements.m_filt_WI101(indices)*1e3-1.0;
        state(end+1:end+sample_num) = measurements.Tg_out_TI102(indices)/295-1.0;
    else
        state(end+1:end+sample_num) = zeros(1,sample_num);
        state(end+1:end+sample_num) = zeros(1,sample_num);
    end

    state(end+1:end+4) = stations_working-0.5;
    % state(end+1) = measurements.t_meas(end);
    state(end+1) = res_solvent*200-0.5;
    
    %check the shape and dimension of the state 
    if isrow(state)
        state = state';  % Transpose to column vector
    end
    if size(state) ~= agent.state_dim
        warning('The dimension of the state is %d, which is not equal to %d. Exiting the program.', size(state), agent.state_dim);
        return;  % Exit the program
    end

    % RL output:
    %   - t_cycle: cycle duraction for the next cycle (in future implementations, RL can be used in controller_online for triggering a cycle switch based on real time measurements)
    %   - V_slurry: slurry volume for the next cycle

    if control_mode == 0

        u.V_slurry=u_nominal.V_slurry;
        action = [u.V_slurry*1e7; u.t_cycle];
    
    elseif control_mode == 1 %training mode

        % select action from actor neural network
        action = gather(agent.select_action(state));
        action = double(extractdata(action));
        % add exploration noise
        action = action + agent.exploration_noise*randn(agent.action_dim,1);
        % scale action from [-1,1] to [min_action, max_action]
        control = 0.5*(action+1.0).*(agent.max_action-agent.min_action) + agent.min_action;
        % action saturation
        control = clip(control, agent.min_action, agent.max_action); 
        % implement the action
        u.V_slurry=control(1)*1e-7;
        u.t_cycle=round(control(2));
   
    elseif control_mode == 2 %testing mode

        action = gather(agent.select_action(state));
        action = double(extractdata(action));
        % scale action from [-1,1] to [min_action, max_action]
        control = 0.5*(action+1.0).*(agent.max_action-agent.min_action) + agent.min_action;

        % implement the action
        u.V_slurry=control(1)*1e-7;
        u.t_cycle=round(control(2));
        
    end

    %reward calculation
    alpha = 1e3; %steepness parameter (adjust as needed)
    penalty =  0.5 * (1 + tanh(alpha*(res_solvent - 0.005))); %quality penalty: res_solvent has to be < 0.005
    reward = u.V_slurry/u.t_cycle*1e7  - 10.0*penalty;
    data_buffer{end}{2} = state; %current state is the next_state for the last cycle
    data_buffer{end+1} = {state, state, action, reward}; %the second state is a place holder


    %% do not modify part below
    if stations_working(1)==0 % if Station 1 is empty at next cycle
       u.V_slurry=0;          % then no slurry loaded in Station 1
    end

    % Store manipulated variables profile
    operating_vars.n_cycle_vector=[operating_vars.n_cycle_vector n_cycle];         
    operating_vars.V_slurry_vector=[operating_vars.V_slurry_vector u.V_slurry];
    if n_cycle > 1
        operating_vars.t_cycle_vector=[operating_vars.t_cycle_vector u.t_cycle];
    end
    
    if round(u.t_cycle)-u.t_cycle>1e-6
       error('u.t_cycle must be an integer!')
    end

end