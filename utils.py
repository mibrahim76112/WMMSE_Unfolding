
import numpy as np
import tensorflow as tf

tf1 = tf.compat.v1

# ---------------------------
# Numpy helpers / baselines
# ---------------------------
# Compute power for bisection search in the optimization of the transmitter precoder 
# - eq. (18) in the paper by Shi et al.
def compute_P(Phi_diag_elements, Sigma_diag_elements, mu):
  nr_of_BS_antennas = Phi_diag_elements.size
  mu_array = mu*np.ones(Phi_diag_elements.size)
  result = np.divide(Phi_diag_elements,(Sigma_diag_elements + mu_array)**2)
  result = np.sum(result)
  return result


def compute_norm_of_complex_array(x):
  result = np.sqrt(np.sum((np.absolute(x))**2))
  return result


def compute_sinr(channel, precoder, noise_power, user_id, selected_users):
    nr_of_users = np.size(channel,0)
    numerator = (np.absolute(np.matmul(np.conj(channel[user_id,:]),precoder[user_id,:])))**2

    inter_user_interference = 0
    for user_index in range(nr_of_users):
      if user_index != user_id and user_index in selected_users:
        inter_user_interference = inter_user_interference + (np.absolute(np.matmul(np.conj(channel[user_id,:]),precoder[user_index,:])))**2
    denominator = noise_power + inter_user_interference

    result = numerator/denominator
    return result


def compute_user_weights(nr_of_users, selected_users):
  result = np.ones(nr_of_users)
  for user_index in range(nr_of_users):
    if not (user_index in selected_users):
      result[user_index] = 0
  return result


def compute_weighted_sum_rate(user_weights, channel, precoder, noise_power, selected_users):
   result = 0
   nr_of_users = np.size(channel,0)
   
   for user_index in range(nr_of_users):
     if user_index in selected_users:
       user_sinr = compute_sinr(channel, precoder, noise_power, user_index, selected_users)
       result = result + user_weights[user_index]*np.log2(1 + user_sinr)
    
   return result


def compute_sinr_nn(channel, precoder, noise_power, user_id, nr_of_users):

    numerator = tf.reduce_sum((tf.matmul(tf.transpose(channel[user_id]),precoder[user_id]))**2)
    inter_user_interference = 0
    for user_index in range(nr_of_users):
      if user_index != user_id:
        inter_user_interference = inter_user_interference +  tf.reduce_sum((tf.matmul(tf.transpose(channel[user_id]),precoder[user_index]))**2)
    denominator = noise_power + inter_user_interference

    result = numerator/denominator
    return result


def compute_WSR_nn(user_weights, channel, precoder, noise_power, nr_of_users, nr_of_samples_per_batch):
   result = 0
   for batch_index in range(nr_of_samples_per_batch):
     for user_index in range(nr_of_users):
        user_sinr = compute_sinr_nn(channel[batch_index], precoder[batch_index], noise_power, user_index, nr_of_users)
        result = result + user_weights[batch_index][user_index] * (
            tf.math.log(1 + user_sinr) / tf.math.log(tf.cast(2.0, tf.float64))
        )
   return result



# Computes a channel realization and returns it in two formats, one for the WMMSE and one for the deep unfolded WMMSE.
# It also returns the initialization value of the transmitter precoder, which is used as input in the computation graph of the deep unfolded WMMSE.
def compute_channel(nr_of_BS_antennas, nr_of_users, total_power, path_loss_option = False, path_loss_range = [-5,5] ):
  channel_nn = []
  initial_transmitter_precoder = []
  channel_WMMSE = np.zeros((nr_of_users, nr_of_BS_antennas)) + 1j*np.zeros((nr_of_users, nr_of_BS_antennas))

  
  for i in range(nr_of_users):

      regularization_parameter_for_RZF_solution = 0
      path_loss = 0 # path loss is 0 dB by default, otherwise it is drawn randomly from a uniform distribution (N.B. it is different for each user)
      if path_loss_option == True:
        path_loss = np.random.uniform(path_loss_range[0],path_loss_range[-1])
        regularization_parameter_for_RZF_solution = regularization_parameter_for_RZF_solution + 1/((10**(path_loss/10))*total_power) # computed as in "MMSE precoding for multiuser MISO downlink transmission with non-homogeneous user SNR conditions" by D.H. Nguyen and T. Le-Ngoc

      result_real = np.sqrt(10**(path_loss/10))*np.sqrt(0.5)*np.random.normal(size = (nr_of_BS_antennas,1))
      result_imag  =  np.sqrt(10**(path_loss/10))*np.sqrt(0.5)*np.random.normal(size = (nr_of_BS_antennas,1))
      channel_WMMSE[i,:] = np.reshape(result_real,(1,nr_of_BS_antennas)) + 1j*np.reshape(result_imag, (1,nr_of_BS_antennas))
      result_col_1 = np.vstack((result_real,result_imag))
      result_col_2 = np.vstack((-result_imag,result_real))
      result =  np.hstack((result_col_1, result_col_2))
      initial_transmitter_precoder.append(result_col_1)
      channel_nn.append(result)

  initial_transmitter_precoder_array = np.array(initial_transmitter_precoder)
  initial_transmitter_precoder_array = np.sqrt(total_power)*initial_transmitter_precoder_array/np.linalg.norm(initial_transmitter_precoder_array)
  initial_transmitter_precoder = []

  for i in range(nr_of_users):
    initial_transmitter_precoder.append(initial_transmitter_precoder_array[i])

  return channel_nn, initial_transmitter_precoder, channel_WMMSE, regularization_parameter_for_RZF_solution


# Computes the zero-forcing solution as in "MMSE precoding for multiuser MISO downlink transmission with non-homogeneous user SNR conditions" by D.H. Nguyen and T. Le-Ngoc
def zero_forcing(channel_realization, total_power):
  
  ZF_solution = np.matmul((np.transpose(channel_realization)),np.linalg.inv(np.matmul(np.conj(channel_realization),(np.transpose(channel_realization)))))
  ZF_solution = ZF_solution*np.sqrt(total_power)/np.linalg.norm(ZF_solution) # scaled according to the power constraint

  return np.transpose(ZF_solution)


# Computes the regularized zero-forcing solution as in "MMSE precoding for multiuser MISO downlink transmission with non-homogeneous user SNR conditions" by D.H. Nguyen and T. Le-Ngoc
def regularized_zero_forcing(channel_realization, total_power, regularization_parameter = 0, path_loss_option = False):
  nr_of_users = channel_realization.shape[0]

  if path_loss_option == False:
    RZF_solution = np.matmul((np.transpose(channel_realization)),np.linalg.inv(np.matmul(np.conj(channel_realization),(np.transpose(channel_realization))) + nr_of_users/total_power*np.eye(nr_of_users, nr_of_users)))
  else:
    RZF_solution = np.matmul((np.transpose(channel_realization)),np.linalg.inv(np.matmul(np.conj(channel_realization),(np.transpose(channel_realization))) + regularization_parameter*np.eye(nr_of_users, nr_of_users)))

  RZF_solution = RZF_solution*np.sqrt(total_power)/np.linalg.norm(RZF_solution) # scaled according to the power constraint

  return np.transpose(RZF_solution)


# Builds one PGD iteration in the deep unfolded WMMSE network
def PGD_step(init, name, mse_weights, user_weights, receiver_precoder, channel,
             initial_transmitter_precoder, total_power,
             nr_of_users, nr_of_BS_antennas, nr_of_samples_per_batch):

  with tf1.variable_scope(name): 

    step_size =  tf.Variable(tf.constant(init, dtype=tf.float64), name="step_size", dtype=tf.float64)

    # First iteration
    a1_exp = tf.tile(tf.expand_dims(mse_weights[:,0,:],-1),[1,2*nr_of_BS_antennas,2*nr_of_BS_antennas])
    a2_exp = tf.tile(tf.expand_dims(user_weights[:,0,:],-1),[1,2*nr_of_BS_antennas,2*nr_of_BS_antennas])
    a3_exp = tf.tile(tf.expand_dims(tf.reduce_sum((receiver_precoder[:,0,:,:])**2,axis =-2),-1),[1,2*nr_of_BS_antennas,2*nr_of_BS_antennas])    
    temp = a1_exp*a2_exp*a3_exp*tf.matmul(channel[:,0,:,:],tf.transpose(channel[:,0,:,:],perm = [0,2,1]))
    
    # Next iterations
    for i in range(1, nr_of_users):
      a1_exp = tf.tile(tf.expand_dims(mse_weights[:,i,:],-1),[1,2*nr_of_BS_antennas,2*nr_of_BS_antennas])
      a2_exp = tf.tile(tf.expand_dims(user_weights[:,i,:],-1),[1,2*nr_of_BS_antennas,2*nr_of_BS_antennas])
      a3_exp = tf.tile(tf.expand_dims(tf.reduce_sum((receiver_precoder[:,i,:,:])**2,axis =-2),-1),[1,2*nr_of_BS_antennas,2*nr_of_BS_antennas])
      temp = temp + a1_exp*a2_exp*a3_exp*tf.matmul(channel[:,i,:,:],tf.transpose(channel[:,i,:,:],perm = [0,2,1]))

    sum_gradient = temp 

    gradient = []

    # Gradient computation
    for i in range(nr_of_users):
      a1_exp = tf.tile(tf.expand_dims(mse_weights[:,i,:],-1),[1,2*nr_of_BS_antennas,1])
      a2_exp = tf.tile(tf.expand_dims(user_weights[:,i,:],-1),[1,2*nr_of_BS_antennas,1])
      gradient.append(step_size*(-2.0*a1_exp*a2_exp*tf.matmul(channel[:,i,:,:],receiver_precoder[:,i,:,:])+ 2*tf.matmul(sum_gradient,initial_transmitter_precoder[:,i,:,:]))) 
      
    tf.stack(gradient)
    gradient = tf.transpose( tf.stack(gradient), perm=[1,0,2,3])
    output_temp = initial_transmitter_precoder - gradient

    output = []
    for i in range(nr_of_samples_per_batch):
      output.append(tf.cond((tf.linalg.norm(output_temp[i]))**2 < total_power, lambda: output_temp[i] , lambda: tf.sqrt(tf.cast(total_power, tf.float64))*output_temp[i]/tf.linalg.norm(output_temp[i]))) 

    return tf.stack(output), step_size



def run_WMMSE(epsilon, channel, selected_users, total_power, noise_power, user_weights, max_nr_of_iterations, log = False):

  nr_of_users = np.size(channel,0)
  nr_of_BS_antennas = np.size(channel,1)
  WSR=[] # to check if the WSR (our cost function) increases at each iteration of the WMMSE
  break_condition = epsilon + 1 # break condition to stop the WMMSE iterations and exit the while
  receiver_precoder = np.zeros(nr_of_users) + 1j*np.zeros(nr_of_users) # receiver_precoder is "u" in the paper of Shi et al. (it's a an array of complex scalars)
  mse_weights = np.ones(nr_of_users) # mse_weights is "w" in the paper of Shi et al. (it's a an array of real scalars)
  transmitter_precoder = np.zeros((nr_of_users, nr_of_BS_antennas)) + 1j*np.zeros((nr_of_users, nr_of_BS_antennas))# transmitter_precoder is "v" in the paper of Shi et al. (it's a complex matrix)
  
  new_receiver_precoder = np.zeros(nr_of_users) + 1j*np.zeros(nr_of_users) # for the first iteration 
  new_mse_weights = np.zeros(nr_of_users) # for the first iteration
  new_transmitter_precoder = np.zeros((nr_of_users, nr_of_BS_antennas)) + 1j*np.zeros((nr_of_users, nr_of_BS_antennas)) # for the first iteration

  
  # Initialization of transmitter precoder
  for user_index in range(nr_of_users):
    if user_index in selected_users:
      transmitter_precoder[user_index,:] = channel[user_index,:]
  transmitter_precoder = transmitter_precoder/np.linalg.norm(transmitter_precoder)*np.sqrt(total_power)
  
  # Store the WSR obtained with the initialized trasmitter precoder    
  WSR.append(compute_weighted_sum_rate(user_weights, channel, transmitter_precoder, noise_power, selected_users))

  # Compute the initial power of the transmitter precoder
  initial_power = 0
  for user_index in range(nr_of_users):
    if user_index in selected_users:
      initial_power = initial_power + (compute_norm_of_complex_array(transmitter_precoder[user_index,:]))**2 
  if log == True:
    print("Power of the initialized transmitter precoder:", initial_power)

  nr_of_iteration_counter = 0 # to keep track of the number of iteration of the WMMSE

  while break_condition >= epsilon and nr_of_iteration_counter<=max_nr_of_iterations:
    
    nr_of_iteration_counter = nr_of_iteration_counter + 1
    if log == True:
      print("WMMSE ITERATION: ", nr_of_iteration_counter)

    # Optimize receiver precoder - eq. (5) in the paper of Shi et al.
    for user_index_1 in range(nr_of_users):
      if user_index_1 in selected_users:
        user_interference = 0.0
        for user_index_2 in range(nr_of_users):
          if user_index_2 in selected_users:
            user_interference = user_interference + (np.absolute(np.matmul(np.conj(channel[user_index_1,:]),transmitter_precoder[user_index_2,:])))**2

        new_receiver_precoder[user_index_1] = np.matmul(np.conj(channel[user_index_1,:]),transmitter_precoder[user_index_1,:]) / (noise_power + user_interference)

    # Optimize mse_weights - eq. (13) in the paper of Shi et al.
    for user_index_1 in range(nr_of_users):
      if user_index_1 in selected_users:

        user_interference = 0 # it includes the channel of all selected users
        inter_user_interference = 0 # it includes the channel of all selected users apart from the current one
        
        for user_index_2 in range(nr_of_users):
          if user_index_2 in selected_users:
            user_interference = user_interference + (np.absolute(np.matmul(np.conj(channel[user_index_1,:]),transmitter_precoder[user_index_2,:])))**2
        for user_index_2 in range(nr_of_users):
          if user_index_2 != user_index_1 and user_index_2 in selected_users:
            inter_user_interference = inter_user_interference + (np.absolute(np.matmul(np.conj(channel[user_index_1,:]),transmitter_precoder[user_index_2,:])))**2
        
        new_mse_weights[user_index_1] = (noise_power + user_interference)/(noise_power + inter_user_interference)

    A = np.zeros((nr_of_BS_antennas,nr_of_BS_antennas))+1j*np.zeros((nr_of_BS_antennas,nr_of_BS_antennas))
    for user_index in range(nr_of_users):
      if user_index in selected_users:
        # hh should be an hermitian matrix of size (nr_of_BS_antennas X nr_of_BS_antennas)
        hh = np.matmul(np.reshape(channel[user_index,:],(nr_of_BS_antennas,1)),np.conj(np.transpose(np.reshape(channel[user_index,:],(nr_of_BS_antennas,1)))))
        A = A + (new_mse_weights[user_index]*user_weights[user_index]*(np.absolute(new_receiver_precoder[user_index]))**2)*hh

    Sigma_diag_elements_true, U = np.linalg.eigh(A)
    Sigma_diag_elements = copy.deepcopy(np.real(Sigma_diag_elements_true))
    Lambda = np.zeros((nr_of_BS_antennas,nr_of_BS_antennas)) + 1j*np.zeros((nr_of_BS_antennas,nr_of_BS_antennas))
    
    for user_index in range(nr_of_users):
      if user_index in selected_users:     
        hh = np.matmul(np.reshape(channel[user_index,:],(nr_of_BS_antennas,1)),np.conj(np.transpose(np.reshape(channel[user_index,:],(nr_of_BS_antennas,1)))))
        Lambda = Lambda + ((user_weights[user_index])**2)*((new_mse_weights[user_index])**2)*((np.absolute(new_receiver_precoder[user_index]))**2)*hh

    Phi = np.matmul(np.matmul(np.conj(np.transpose(U)),Lambda),U)
    Phi_diag_elements_true = np.diag(Phi)
    Phi_diag_elements = copy.deepcopy(Phi_diag_elements_true)
    Phi_diag_elements = np.real(Phi_diag_elements)

    for i in range(len(Phi_diag_elements)):
      if Phi_diag_elements[i]<np.finfo(float).eps:
        Phi_diag_elements[i] = np.finfo(float).eps
      if (Sigma_diag_elements[i])<np.finfo(float).eps:
        Sigma_diag_elements[i] = 0

    # Check if mu = 0 is a solution (eq.s (15) and (16) of in the paper of Shi et al.)
    power = 0 # the power of transmitter precoder (i.e. sum of the squared norm)
    for user_index in range(nr_of_users):
      if user_index in selected_users:
        if np.linalg.det(A) != 0:
          w = np.matmul(np.linalg.inv(A),np.reshape(channel[user_index,:],(nr_of_BS_antennas,1)))*user_weights[user_index]*new_mse_weights[user_index]*(new_receiver_precoder[user_index])
          power = power + (compute_norm_of_complex_array(w))**2

    # If mu = 0 is a solution, then mu_star = 0
    if np.linalg.det(A) != 0 and power <= total_power:
      mu_star = 0
    # If mu = 0 is not a solution then we search for the "optimal" mu by bisection
    else:
      power_distance = [] # list to store the distance from total_power in the bisection algorithm 
      mu_low = np.sqrt(1/total_power*np.sum(Phi_diag_elements))
      mu_high = 0
      low_point = compute_P(Phi_diag_elements, Sigma_diag_elements, mu_low)
      high_point = compute_P(Phi_diag_elements, Sigma_diag_elements, mu_high)

      obtained_power = total_power + 2*power_tolerance # initialization of the obtained power such that we enter the while 

      # Bisection search
      while np.absolute(total_power - obtained_power) > power_tolerance:
        mu_new = (mu_high + mu_low)/2
        obtained_power = compute_P(Phi_diag_elements, Sigma_diag_elements, mu_new) # eq. (18) in the paper of Shi et al.
        power_distance.append(np.absolute(total_power - obtained_power))
        if obtained_power > total_power:
          mu_high = mu_new
        if obtained_power < total_power:
          mu_low = mu_new
      mu_star = mu_new
      if log == True:
        print("first value:", power_distance[0])
        plt.title("Distance from the target value in bisection (it should decrease)")
        plt.plot(power_distance)
        plt.show()

    for user_index in range(nr_of_users):
      if user_index in selected_users:
        new_transmitter_precoder[user_index,:] = np.matmul(np.linalg.inv(A + mu_star*np.eye(nr_of_BS_antennas)),channel[user_index,:])*user_weights[user_index]*new_mse_weights[user_index]*(new_receiver_precoder[user_index]) 

    # To select only the weights of the selected users to check the break condition
    mse_weights_selected_users = []
    new_mse_weights_selected_users = []
    for user_index in range(nr_of_users): 
      if user_index in selected_users:
        mse_weights_selected_users.append(mse_weights[user_index])
        new_mse_weights_selected_users.append(new_mse_weights[user_index])

    mse_weights = deepcopy(new_mse_weights)
    transmitter_precoder = deepcopy(new_transmitter_precoder)
    receiver_precoder = deepcopy(new_receiver_precoder)

    WSR.append(compute_weighted_sum_rate(user_weights, channel, transmitter_precoder, noise_power, selected_users))
    break_condition = np.absolute(np.log2(np.prod(new_mse_weights_selected_users))-np.log2(np.prod(mse_weights_selected_users)))

  if log == True:
    plt.title("Change of the WSR at each iteration of the WMMSE (it should increase)")
    plt.plot(WSR,'bo')
    plt.show()

  return transmitter_precoder, receiver_precoder, mse_weights, WSR[-1]

