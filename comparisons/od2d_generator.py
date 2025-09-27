import pickle
import time

import orbit_defender2d.utils.utils as U
import orbit_defender2d.king_of_the_hill.pettingzoo_env as PZE
from orbit_defender2d.king_of_the_hill import default_game_parameters as DGP

if __name__ == "__main__":
    start_time = time.time()
    penv = PZE.parallel_env()

    experiences_data = []
    mobile_fuel_data = []
    seeker_fuel_data = []
    termination_data = []
    NUM_EPISODES = 1_000

    for epi_num in range(NUM_EPISODES):
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

    with open('experiences_data_od2d.pkl', 'wb') as f:
        pickle.dump(experiences_data, f)

    with open('mobile_fuel_data_od2d.pkl', 'wb') as f:
        pickle.dump(mobile_fuel_data, f)

    with open('seeker_fuel_data_od2d.pkl', 'wb') as f:
        pickle.dump(seeker_fuel_data, f)

    with open('termination_data_od2d.pkl', 'wb') as f:
        pickle.dump(termination_data, f)
