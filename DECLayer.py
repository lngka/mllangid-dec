from tensorflow.keras.layers import Layer, InputSpec
import tensorflow.keras.backend as K
import tensorflow as tf


class DECLayer(Layer):
    """
    Clustering layer converts input sample (feature) to a vector that represents the probability of the
    sample belonging to each cluster.

    # Arguments
        n_lang: number of language cluster.
        weights: shape (n_lang, n_features) witch represents the initial cluster centers.
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
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DECLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        '''This method can be used to create weights that
        depend on the shape(s) of the input(s), using `add_weight()`. `__call__()`
        will automatically build the layer (if it has not been built yet) by
        calling `build()`
        '''
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))

        self.clusters = self.add_weight(shape=(
            self.n_clusters, int(input_dim)), initializer='glorot_uniform', name='clusters')

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

        minus_centroids = inputs_expanded - self.clusters

        squared = K.square(minus_centroids)

        squared_sum = K.sum(squared, axis=2)

        divide_alpha = squared_sum / self.alpha

        # the numnerator in q_ịj formular in the paper
        numerator = 1.0 / (1.0 + divide_alpha)
        numerator **= (self.alpha + 1.0) / 2.0

        denominator = K.sum(numerator, axis=1)

        quiu = K.transpose(numerator) / denominator
        quiu = K.transpose(quiu)

        # print('inputs: ', inputs.shape)
        # print('inputs_expanded: ', inputs_expanded.shape)
        # print('minus_centroids: ', minus_centroids.shape)
        # print('squared: ', squared.shape)
        # print('squared_sum: ', squared_sum.shape)
        # print('divide_alpha: ', divide_alpha.shape)
        # print('numerator: ', numerator.shape)
        # print('denominator: ', denominator.shape)
        # print('quiu: ', quiu)

        # q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs,
        #                                                axis=1) - self.clusters), axis=2) / self.alpha))
        # q **= (self.alpha + 1.0) / 2.0
        # q = K.transpose(K.transpose(q) / K.sum(q, axis=1))

        return quiu

    def get_config(self):
        '''
        Returns a dictionary containing the configuration used
        to initialize this layer. If the keys differ from the arguments
        in `__init__`, then override `from_config(self)` as well.
        This method is used when saving the layer or a model that contains this layer.
        '''
        config = {'n_clusters': self.n_clusters}
        base_config = super(DECLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters
