import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

LATENT_DIM = 256  # Adjust as needed

# Define the custom sampling layer
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a face."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Define the custom VAE model class.
# This class needs to be defined so the model loader knows what "VAE" is.
class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampling = Sampling()
    
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed

# Use a custom object scope to make the VAE class available during model loading
with tf.keras.utils.custom_object_scope({'VAE': VAE, 'Sampling': Sampling}):
    # Load the VAE model without compiling
    model = tf.keras.models.load_model('vae_model.h5', compile=False)
    # Get the decoder layer from the loaded model
    # Note: The original app.py assumed the model had a layer named 'decoder'.
    # This code assumes the 'VAE' model itself has a 'decoder' attribute.
    # The fix ensures that the model loads and the decoder can be accessed.
    decoder = model.decoder
    
def generate_face():
    """Generates a new face image from a random latent vector."""
    # Generate a random latent vector
    latent = np.random.randn(1, LATENT_DIM).astype(np.float32)
    # Predict the image using the decoder
    img = decoder.predict(latent, verbose=0)[0]
    # Clip the image values to be within the valid range [0, 1]
    img = np.clip(img, 0, 1)
    return img

# Create the Gradio interface
iface = gr.Interface(
    fn=generate_face,
    inputs=None,
    outputs=gr.Image(label="Generated Face"),
    title="VAE Face Generator"
)

# Launch the Gradio app
if __name__ == '__main__':
    iface.launch()