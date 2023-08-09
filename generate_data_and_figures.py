from generate_data import test_accuracy, test_indifference_to_initial_condition, K_most_likely_data,generate_traj_for_fig2
from code_for_figures import figure_2, figure_3, figure_5, figure_6, figure_7

N_REPETITIONS = 20 # In the paper we used 1000. Here we put a smaller number for demonstration

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
