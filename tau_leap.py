import numpy as np


TAU_DAY = 1 / 3  # Time step for daytime tau leap
TAU_NIGHT = 2 / 3  # Time step for night time tau leap


def SIR_tau_leap(population, movement, mov_ratio, initial_cond, beta):
    n = len(population)
    # initialize the result array
    result = np.zeros((n, 3, 2))
    result[:, :, 0] = initial_cond

    # start the tau leap for the day time
    # separate the movers in S,I,R compartments with movement matrix
    # simulate the transmission among movers
    # first compute the in movement of people in each compartment
    mov_S = mov_ratio @ np.diag(result[:, 0, 0])
    mov_I = mov_ratio @ np.diag(result[:, 1, 0])
    # compute the beta for each location at time i

    # then compute the transmitted people from S to I
    # with the transmission rate being beta_i, which is the destination of the movement
    # movement is the total population movement
    mov_SI = np.divide(np.multiply(mov_I, mov_S), movement)
    mov_SI[~np.isfinite(mov_SI)] = 0
    # make the diagonal of the matrix 0
    mov_SI[range(n), range(n)] = 0
    # tau leap for the day time movement infection force

    transfer_SI = np.random.poisson(np.diag(beta) @ mov_SI * TAU_DAY)

    # update the S,I,R compartments with the local transmission rate
    # extract the S,I,R compartments
    # subtract the movers
    # last term mov_ratio@result[:,0,i-1] records the number of sus/infected people that are moving in the destination location
    S = result[:, 0, 0] - np.sum(mov_S, axis=0).T + mov_ratio @ result[:, 0, 0]
    I = result[:, 1, 0] - np.sum(mov_I, axis=0).T + mov_ratio @ result[:, 1, 0]
    R = result[:, 2, 0]

    # generate a poisson random number for each time step
    force_of_infection = np.random.poisson(TAU_DAY * beta * S * I / population)
    result[:, 0, 1] = S - force_of_infection
    # recover rate is 0.2 per day
    force_of_recovery = np.random.poisson(0.2 * I * TAU_DAY)
    result[:, 1, 1] = I + force_of_infection - force_of_recovery
    result[:, 2, 1] = R + force_of_recovery

    # start the tau leap for the night time
    # add the newly infected people back to their home location
    result[:, 0, 1] = (
        result[:, 0, 1]
        + np.sum(mov_S, axis=0).T
        - np.sum(transfer_SI, axis=0).T
        - mov_ratio @ result[:, 0, 0]
    )
    result[:, 1, 1] = (
        result[:, 1, 1]
        + np.sum(mov_I, axis=0).T
        + np.sum(transfer_SI, axis=0).T
        - mov_ratio @ result[:, 1, 0]
    )
    # find the negative values in result[:,1,i]
    # The reason for the negative values is that the force of infection is too high
    # find the index of the negative values in S
    neg_index_S = np.where(result[:, 0, 1] < 0)
    # if neg_index_S is not empty
    if neg_index_S[0].size != 0:
        # save the negative values
        neg_value_S = result[neg_index_S, 0, 1]
        # set the negative values to 0
        result[neg_index_S, 0, 1] = 0
        # add the negative values to the infected compartment
        result[neg_index_S, 1, 1] = result[neg_index_S, 1, 1] + neg_value_S

    # find the index of the negative values in I
    neg_index_I = np.where(result[:, 1, 1] < 0)
    # if neg_index is not empty
    if neg_index_I[0].size != 0:
        # save the negative values
        neg_valu_I = result[neg_index_I, 1, 1]
        # set the negative values to 0
        result[neg_index_I, 1, 1] = 0
        # add the negative values to the recovered compartment
        result[neg_index_I, 2, 1] = result[neg_index_I, 2, 1] + neg_valu_I
        # update the S,I,R compartments with the local transmission rate
    # extract the S,I,R compartments
    S = result[:, 0, 1]
    I = result[:, 1, 1]
    R = result[:, 2, 1]
    # generate a poisson random number for each time step
    force_of_infection = np.random.poisson(TAU_NIGHT * beta * S * I / population)
    result[:, 0, 1] = S - force_of_infection
    # recover rate is 0.2 per day
    force_of_recovery = np.random.poisson(0.2 * I * TAU_NIGHT)
    result[:, 1, 1] = I + force_of_infection - force_of_recovery
    result[:, 2, 1] = R + force_of_recovery

    # find the negative values in result[:,1,i]
    # The reason for the negative values is that the force of infection is too high
    # find the index of the negative values in S
    neg_index_S = np.where(result[:, 0, 1] < 0)
    # if neg_index_S is not empty
    if neg_index_S[0].size != 0:
        # save the negative values
        neg_value_S = result[neg_index_S, 0, 1]
        # set the negative values to 0
        result[neg_index_S, 0, 1] = 0
        # add the negative values to the infected compartment
        result[neg_index_S, 1, 1] = result[neg_index_S, 1, 1] + neg_value_S

    # find the index of the negative values in I
    neg_index_I = np.where(result[:, 1, 1] < 0)
    # if neg_index is not empty
    if neg_index_I[0].size != 0:
        # save the negative values
        neg_valu_I = result[neg_index_I, 1, 1]
        # set the negative values to 0
        result[neg_index_I, 1, 1] = 0
        # add the negative values to the recovered compartment
        result[neg_index_I, 2, 1] = result[neg_index_I, 2, 1] + neg_valu_I

        # sum the compartments to check if the total population is conserved
    # return nothing for now
    # the first dimension of the result is the location
    # the second dimension is the S,I,R compartments
    # the third dimension is the time step, it has two values, 0 for the beginning of the day, 1 for the end of the day

    return result
