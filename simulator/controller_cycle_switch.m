function [u,u_nominal,operating_vars] = controller_cycle_switch(process_time,cycle_time,...
               stations_working,u,u_nominal,cryst_output_nominal,measurements,operating_vars,x_estim,...
               n_cycle,control_mode, res_solvent, agent)
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
    u.Tinlet_drying
    u.P_compr
    mean(measurements.c_slurry_AI101)
    if length(measurements.m_filt_WI101) > 1
        interp1(measurement.t_meas,measurements.m_filt_WI101,linspace(0,measurements.t_meas(end),30));
        interp1(measurement.t_meas,measurements.Tg_out_TI102,linspace(0,measurements.t_meas(end),30));
    else
        zeros(1,30)
        zeros(1,30)
    end
    stations_working
    measurements.t_meas(end)

%     state=

    res_solvent

    % RL output:
    %   - t_cycle: cycle duraction for the next cycle (in future implementations, RL can be used in controller_online for triggering a cycle switch based on real time measurements)
    %   - V_slurry: slurry volume for the next cycle
    
%     state = TBD;

    if control_mode == 0
        u.V_slurry=u_nominal.V_slurry;
    
    elseif control_mode == 1 %training mode

        % select action from actor neural network
        action = extractdata(agent.select_action(state));
        % add exploration noise
        action = action + agent.exploration_noise*randn(agent.action_dim,1);
        % action saturation
        action = clip(action, agent.min_action, agent.max_action);


        % quality penalty: res_solvent has to be < 0.005
        alpha = 1e4;  % Steepness parameter (adjust as needed)
        penalty =  0.5 * (1 + tanh(alpha * (res_solvent - 0.005)));
        reward = u.V_slurry/u.t_cycle-penalty;
        done = TBD;
        
        agent.replay_buffer = agent.replay_buffer.push({agent.state_pre,...
            state, agent.action_pre, agent.reward_pre, double(agent.done_pre)});
        
        % update agent's memory on the info of the last step
        agent.state_pre = state;
        agent.action_pre = action;
        agent.reward_pre = reward;
        agent.done_pre = done;
        
        % update agent after one episode
        if done %one episode ends
            agent = agent.update();

            %reset memory
            agent.state_pre = zeros(state_dim,1);
            agent.action_pre = zeros(action_dim,1);
            agent.reward_pre = 0;
            agent.done_pre = 0;
            
            % save the learned neural networks
            agent.save()
        end

        % implement the action
        u.V_slurry=action(1)*1e-7;
        u.t_cycle=round(action(2));

    elseif control_mode == 2 %testing mode
        action = extractdata(agent.select_action(state));
        reward = TBD;

        % implement the action
        u.V_slurry=action(1)*1e-6;
        u.t_cycle=round(action(2));
    end

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