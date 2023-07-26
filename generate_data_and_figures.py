from generate_data import test_accuracy, test_indifference_to_initial_condition, K_most_likely_data,generate_traj_for_fig2
from code_for_figures import figure_1, figure_2, figure_3, figure_5, figure_6, figure_7

N_REPETITIONS = 1000 # Lower this number for faster simulations.

if __name__ == '__main__':
    test_indifference_to_initial_condition(N_init_conditions=N_REPETITIONS, selected_regime=0)
    test_accuracy(N_realizations=N_REPETITIONS)
    K_most_likely_data()
    generate_traj_for_fig2()
    # figure_1() #this requires experimental data
    figure_2()
    figure_3()
    figure_5()
    figure_6()
    figure_7()
