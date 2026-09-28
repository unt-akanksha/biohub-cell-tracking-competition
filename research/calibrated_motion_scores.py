"""Optional exact zero-weight short circuit; independent of array library."""


def calibrated_motion_scores(prior, neural_predict, alpha, beta, *, skip_zero_neural=False):
    if type(skip_zero_neural) is not bool:
        raise ValueError('Explicit boolean execution option required')
    if skip_zero_neural and alpha == 0.:
        return beta*prior
    return alpha*neural_predict()+beta*prior
