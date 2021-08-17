import tensorflow as tf

class WackyAttentionModule(tf.keras.models.Model):

    def __init__(
            self,
            embedding_input_dim,
            embedding_output_dim,
    ):

        super(WackyAttentionModule, self).__init__()

        # Based on https://keras.io/api/layers/attention_layers/attention/:

        # Variable-length int sequences.
        query_input = tf.keras.Input(shape=(None,), dtype='int32')
        value_input = tf.keras.Input(shape=(None,), dtype='int32')

        # Embedding lookup.
        token_embedding = tf.keras.layers.Embedding(input_dim=embedding_input_dim, output_dim=embedding_output_dim)
        # Query embeddings of shape [batch_size, Tq, dimension].
        query_embeddings = token_embedding(query_input)
        # Value embeddings of shape [batch_size, Tv, dimension].
        value_embeddings = token_embedding(value_input)

        # CNN layer.
        cnn_layer = tf.keras.layers.Conv1D(
            filters=100,
            kernel_size=4,
            # Use 'same' padding so outputs have the same shape as inputs.
            padding='same')
        # Query encoding of shape [batch_size, Tq, filters].
        query_seq_encoding = cnn_layer(query_embeddings)
        # Value encoding of shape [batch_size, Tv, filters].
        value_seq_encoding = cnn_layer(value_embeddings)

        # Query-value attention of shape [batch_size, Tq, filters].
        query_value_attention_seq = tf.keras.layers.Attention()(
            [query_seq_encoding, value_seq_encoding])

        # Reduce over the sequence axis to produce encodings of shape
        # [batch_size, filters].
        query_encoding = tf.keras.layers.GlobalAveragePooling1D()(
            query_seq_encoding)
        query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
            query_value_attention_seq)

        # Concatenate query and document encodings to produce a DNN input layer.
        self.input_layer = tf.keras.layers.Concatenate()(
            [query_encoding, query_value_attention])


    def call(self, inputs):
