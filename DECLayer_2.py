from tensorflow.keras.layers import Layer, InputSpec, Lambda
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np


class MDECLayer(Layer):
    """
    Clustering layer converts input sample (feature) to a vector that represents the probability of the
    sample belonging to each cluster.

    # Arguments
        n_lang: number of language clusters.
        weights:[means, inv_covmat]
                means.shape = (n_lang, n_features)
                inv_covmat.shape = (n_lang, n_features, n_features)
                Used to calculate the robust mahalanobis distance
        alpha: alpha parameter of Student's t-distribution
    # Input shape
        (batch_size, n_features)
    # Output shape
        Numpy array with shape (batch_size, n_lang), probability of sample belonging to  a language class
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        '''Defines custom layer attributes, and creates layer state
        variables that do not depend on input shapes, using `add_weight()`
        '''
        self.n_clusters = n_clusters
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)
        self.alpha = alpha

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(MDECLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        '''This method can be used to create weights that
        depend on the shape(s) of the input(s), using `add_weight()`. `__call__()`
        will automatically build the layer (if it has not been built yet) by
        calling `build()`
        '''
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))

        self.robust_mean = self.add_weight(shape=(
            self.n_clusters, int(input_dim)), initializer='glorot_uniform', name='robust_mean')

        self.inv_covmat = self.add_weight(shape=(
            self.n_clusters, int(input_dim), int(input_dim)), initializer='glorot_uniform', name='inv_covmat')

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ performs the logic of applying the layer to the input
        student t-distribution:
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_lang)
        """
        inputs_expanded = K.expand_dims(inputs, axis=1)
        #print('inputs_expanded', inputs_expanded.shape)
        #print('robust_mean', self.robust_mean.shape)

        x_minus_mu = inputs_expanded - self.robust_mean
        #print('x_minus_mu', x_minus_mu.shape)

        left_term = list()
        for i in range(self.n_clusters):
            x = x_minus_mu[:, i, :]
            left = K.dot(x, self.inv_covmat[i])
            if len(left_term) == 0:
                left_term = left
            else:
                left_term = K.stack([left_term, left], axis=1)
        #print('inv_covmat', self.inv_covmat.shape)
        #print('left_term', left_term.shape)

        left_term_T = K.permute_dimensions(left_term, (1, 0, 2))
        x_minus_mu_T = K.permute_dimensions(x_minus_mu, (1, 0, 2))
        #print('x_minus_mu_T', x_minus_mu_T.shape)
        #print('left_term_T', left_term_T.shape)

        mahal = K.batch_dot(left_term_T, x_minus_mu_T, axes=[2, 2])
        #print('mahal', mahal.shape)

        mahal_diagonal = list()
        for i in range(self.n_clusters):
            m = mahal[i, :, :]
            diagonal = tf.linalg.tensor_diag_part(m)
            if len(mahal_diagonal) == 0:
                mahal_diagonal = diagonal
            else:
                mahal_diagonal = K.stack([mahal_diagonal, diagonal], axis=1)
        #print('mahal_diagonal', mahal_diagonal.shape)

        md = K.sqrt(mahal_diagonal)
        #print('md', md.shape)

        divide_alpha = md / self.alpha

        # the numnerator in q_ịj formular in the paper
        numerator = 1.0 / (1.0 + divide_alpha)
        numerator **= (self.alpha + 1.0) / 2.0

        denominator = K.sum(numerator, axis=1)

        quiu = K.transpose(numerator) / denominator
        quiu = K.transpose(quiu)

        #print('quiu', quiu.shape)

        return quiu

    def get_config(self):
        '''
        Returns a dictionary containing the configuration used
        to initialize this layer. If the keys differ from the arguments
        in `__init__`, then override `from_config(self)` as well.
        This method is used when saving the layer or a model that contains this layer.
        '''
        config = {'n_clusters': self.n_clusters}
        base_config = super(MDECLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters
