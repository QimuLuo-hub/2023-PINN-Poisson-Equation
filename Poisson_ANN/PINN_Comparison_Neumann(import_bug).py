from PINN_Neumann import *
import pandas as pd


#epochs
epochs_list = [1000, 200, 30000, 40000, 50000]

l2_error_list_epochs = np.zeros(len(epochs_list))
time_list_epochs = np.zeros(len(epochs_list))

#points
N_b_list = [100, 200]
N_f_list = [2000, 4000]

combined_list_points = [(x, y) for x in N_b_list for y in N_f_list]

l2_error_list_points = np.zeros((len(combined_list_points)))

time_list_points = np.zeros((len(combined_list_points)))


#neurons and layers
neurons_list = [20, 60, 64, 120]
hidden_layers_list = [1, 2, 3, 4, 5]

combined_list_neural = [(x, y) for x in neurons_list for y in hidden_layers_list]
l2_error_list_neural = np.zeros((len(combined_list_neural)))
time_list_neural = np.zeros((len(combined_list_neural)))



while True:
    print('for 1 and 2: PINNnet = PINN(2, neurons = 64, hidden_layers = 3, 1)')
    print("1.compare number of data points (N_b, N_f): ", 'epochs: ', epochs_list[1])
    #print("2.compare epochs: ", epochs_list, '(N_b, N_f): ', N_b_list[2], N_f_list[0])
    #print('3.compare neural network structure: ', 'epochs: ', epochs_list[1], '(N_b, N_f): ', N_b_list[2], N_f_list[0])
    print('4.Stop')
    option = input('Choose 1, 2, 3, 4: ')
    choice = ['1', '2', '3', '4']

    if option == choice[0]:
        for k in range(len(combined_list_points)):
            PINNnet = PINN(2, neurons_list[2], hidden_layers_list[2], 1)
            loss, l2, u_pred, l2_error, run_time = train(epochs_list[1], combined_list_points[k][0], combined_list_points[k][1])
            l2_error_list_points[k] = l2_error
            time_list_points[k] = run_time
            # draw(epochs_list[1], 50)
            print('interation: ', epochs_list[1], 'boundary: ', combined_list_points[k][0], 'Interior: ', combined_list_points[k][1], '\n',
                          'run time', run_time, 'l2 error', l2_error)
        df = pd.DataFrame({'Points': combined_list_points, 'Error': l2_error_list_points, 'Runtime': time_list_points})
        '''convert data type'''
        df['Error'] = df['Error'].apply(lambda x: f"{x:.2e}")
        df['Runtime'] = df['Runtime'].apply(lambda x: f"{x:.3f}")  # （Keep 2 decimal place）
        print(df)


    elif option == choice[1]:
        for j in range(len(epochs_list)):
            PINNnet = PINN(2, neurons_list[2], hidden_layers_list[2], 1)
            loss, l2, u_pred, l2_error, run_time = train(epochs_list[j], N_b_list[2], N_f_list[0])
            l2_error_list_epochs[j] = l2_error
            time_list_epochs[j] = run_time
            print('j=', j, 'iteration is', epochs_list[j], '\n', 'runtime is', time_list_epochs[j])

            # draw(epochs_list[j], h)

        # Table for epochs
        df = pd.DataFrame({'Iterations': epochs_list, 'Error': l2_error_list_epochs, 'Runtime': time_list_epochs})
        '''convert data type'''
        df['Error'] = df['Error'].apply(lambda x: f"{x:.2e}")
        df['Runtime'] = df['Runtime'].apply(lambda x: f"{x:.3f}")  # （Keep 2 decimal place）
        print(df)

    elif option == choice[2]:
        for a in range(len(combined_list_neural)):
            PINNnet = PINN(2, combined_list_neural[a][0], combined_list_neural[a][1], 1)
            loss, l2, u_pred, l2_error, run_time = train(epochs_list[1], N_b_list[2], N_f_list[0])
            l2_error_list_neural[a] = l2_error
            time_list_neural[a] = run_time
            print('interation: ', epochs_list[1], 'neurons: ', combined_list_neural[a][0], 'hiddenlayers: ', combined_list_neural[a][1], '\n',
                          'run time', run_time, 'l2 error', l2_error)
            draw(epochs_list[1], 50)
        df = pd.DataFrame({'Neural structure(neurons, layers)': combined_list_neural, 'Error': l2_error_list_neural, 'Runtime': time_list_neural})
        '''convert data type'''
        df['Error'] = df['Error'].apply(lambda x: f"{x:.3e}")
        df['Runtime'] = df['Runtime'].apply(lambda x: f"{x:.3f}")  # （Keep 2 decimal place）
        print(df)

    elif option == choice[3]:
        break

    else:
        print('You can only enter 1, 2, 3')






