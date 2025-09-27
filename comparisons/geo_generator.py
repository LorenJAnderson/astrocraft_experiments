import pickle
import time

import numpy as np

import orbit_defender2d.utils.utils as U
from orbit_defender2d.king_of_the_hill import koth
import orbit_defender2d.king_of_the_hill.pettingzoo_env as PZE
import game_parameters_default as DGP
import game_parameters_case1 as GP_1
import game_parameters_case2 as GP_2
import game_parameters_case3 as GP_3
import game_parameters_case4 as GP_4
import game_parameters_case5 as GP_5

if __name__ == "__main__":
    start_time = time.time()

    experiences_data = []
    mobile_fuel_data = []
    seeker_fuel_data = []
    termination_data = []
    NUM_EPISODES = 1_000

    for epi_num in range(NUM_EPISODES):
        game_parameters = np.random.randint(6)
        if game_parameters == 1:
            GP = GP_1
        elif game_parameters == 2:
            GP = GP_2
        elif game_parameters == 3:
            GP = GP_3
        elif game_parameters == 4:
            GP = GP_4
        elif game_parameters == 5:
            GP = GP_5
        else:
            GP = DGP

        GAME_PARAMS = koth.KOTHGameInputArgs(
            max_ring=GP.MAX_RING,
            min_ring=GP.MIN_RING,
            geo_ring=GP.GEO_RING,
            init_board_pattern_p1=GP.INIT_BOARD_PATTERN_P1,
            init_board_pattern_p2=GP.INIT_BOARD_PATTERN_P2,
            init_fuel=GP.INIT_FUEL,
            init_ammo=GP.INIT_AMMO,
            min_fuel=GP.MIN_FUEL,
            fuel_usage=GP.FUEL_USAGE,
            engage_probs=GP.ENGAGE_PROBS,
            illegal_action_score=GP.ILLEGAL_ACT_SCORE,
            in_goal_points=GP.IN_GOAL_POINTS,
            adj_goal_points=GP.ADJ_GOAL_POINTS,
            fuel_points_factor=GP.FUEL_POINTS_FACTOR,
            win_score=GP.WIN_SCORE,
            max_turns=GP.MAX_TURNS,
            fuel_points_factor_bludger=GP.FUEL_POINTS_FACTOR_BLUDGER,
        )
        penv = PZE.parallel_env(game_params=GAME_PARAMS, training_randomize=False, plr_aliases=None)

        epi_experiences = []
        epi_mobile_fuel = []
        epi_alpha_seeker_fuel = []
        epi_beta_seeker_fuel = []
        if (epi_num % 100) == 0:
            print('Episode: ', epi_num)
        first_obs, first_info = penv.reset()
        epi_experiences.append((first_obs, None, None, first_info, None))

        step_mobile_fuel = []
        for unit_id in range(1, 11):
            step_mobile_fuel.append(penv.kothgame.game_state[U.P1][U.TOKEN_STATES][unit_id].satellite.fuel)
            step_mobile_fuel.append(penv.kothgame.game_state[U.P2][U.TOKEN_STATES][unit_id].satellite.fuel)
        epi_mobile_fuel.append(step_mobile_fuel)
        epi_alpha_seeker_fuel.append(penv.kothgame.game_state[U.P1][U.TOKEN_STATES][0].satellite.fuel)
        epi_beta_seeker_fuel.append(penv.kothgame.game_state[U.P2][U.TOKEN_STATES][0].satellite.fuel)

        while True:
            actions = penv.kothgame.get_random_valid_actions()
            penv.actions = actions
            encoded_actions = penv.encode_all_discrete_actions(actions=actions)
            observations, rewards, dones, info = penv.step(actions=encoded_actions)
            epi_experiences.append((observations, rewards, dones, info, encoded_actions))

            step_mobile_fuel = []
            for unit_id in range(1, 11):
                step_mobile_fuel.append(penv.kothgame.game_state[U.P1][U.TOKEN_STATES][unit_id].satellite.fuel)
                step_mobile_fuel.append(penv.kothgame.game_state[U.P2][U.TOKEN_STATES][unit_id].satellite.fuel)
            epi_mobile_fuel.append(step_mobile_fuel)
            epi_alpha_seeker_fuel.append(penv.kothgame.game_state[U.P1][U.TOKEN_STATES][0].satellite.fuel)
            epi_beta_seeker_fuel.append(penv.kothgame.game_state[U.P2][U.TOKEN_STATES][0].satellite.fuel)

            if any([dones[d] for d in dones.keys()]):
                break

        alpha_score = penv.kothgame.game_state[U.P1][U.SCORE]
        beta_score = penv.kothgame.game_state[U.P2][U.SCORE]
        result = 'draw' if alpha_score == beta_score else 'victory'
        experiences_data.append((epi_experiences, result))
        mobile_fuel_data.append(epi_mobile_fuel)
        seeker_fuel_data.append((epi_alpha_seeker_fuel, epi_beta_seeker_fuel))

        if penv.kothgame.game_state[U.P1][U.TOKEN_STATES][0].satellite.fuel <= DGP.MIN_FUEL:
            term_cond = "alpha seeker out of fuel"
        elif penv.kothgame.game_state[U.P2][U.TOKEN_STATES][0].satellite.fuel <= DGP.MIN_FUEL:
            term_cond = "beta seeker out of fuel"
        elif alpha_score >= DGP.WIN_SCORE:
            term_cond = "alpha reached Win Score"
        elif beta_score >= DGP.WIN_SCORE:
            term_cond = "beta reached Win Score"
        elif penv.kothgame.game_state[U.TURN_COUNT] >= DGP.MAX_TURNS:
            term_cond = "max turns reached"
        else:
            term_cond = "unknown"
        termination_data.append(term_cond)

    end_time = time.time()
    print('Elapsed time: ', end_time - start_time, 'seconds.')

    with open('experiences_data_patrol.pkl', 'wb') as f:
        pickle.dump(experiences_data, f)

    with open('mobile_fuel_data_patrol.pkl', 'wb') as f:
        pickle.dump(mobile_fuel_data, f)

    with open('seeker_fuel_data_patrol.pkl', 'wb') as f:
        pickle.dump(seeker_fuel_data, f)

    with open('termination_data_patrol.pkl', 'wb') as f:
        pickle.dump(termination_data, f)
