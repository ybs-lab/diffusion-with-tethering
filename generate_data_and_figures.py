from generate_data import test_accuracy, test_indifference_to_initial_condition, K_most_likely_data
from code_for_figures import figure_1, figure_2, figure_3, figure_5, figure_6, figure_7

if __name__ == '__main__':
    test_indifference_to_initial_condition(N_init_conditions=1000, selected_regime=0)
    test_accuracy(N_realizations=1000)
    K_most_likely_data()
    figure_1()
    figure_2()
    figure_3()
    figure_5()
    figure_6()
    figure_7()
