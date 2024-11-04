embedding_dim = 100
noise_dim = 100
image_dim = (128,128,3)
text_embedding_dim = 100
batch_size = 32

#Building a Generator
def build_generator(vocab_size, embedding_dim, noise_dim, max_length):
# Text input
text_input = Input(shape=(max_length,), name="text_input")
text_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, 
input_length=max_length)(text_input)
text_embedding = Flatten()(text_embedding)
text_dense = Dense(256)(text_embedding)
text_dense = Activation('relu')(text_dense)
# Noise input
noise_input = Input(shape=(noise_dim,), name="noise_input")
noise_dense = Dense(256)(noise_input)
noise_dense = Activation('relu')(noise_dense)
# Combine text embedding and noise
combined_input = Concatenate()([text_dense, noise_dense])
# Further dense layer
x = Dense(128 * 8 * 8)(combined_input)
x = Reshape((8, 8, 128))(x)
# Upsampling to generate the image
x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same')(x)
output = Activation('tanh')(x)
model = Model([text_input, noise_input], output)
return model
# Create generator
generator = build_generator(vocab_size, embedding_dim, noise_dim, max_length)
generator.summary()

#Building a Disciminator
def build_discriminator(image_dim, vocab_size, embedding_dim, max_length):
# Image input
img_input = Input(shape=image_dim, name="image_input")
# Text input
text_input = Input(shape=(max_length,), name="text_input")
text_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, 
input_length=max_length)(text_input)
text_embedding = Flatten()(text_embedding)
text_dense = Dense(np.prod(image_dim))(text_embedding)
text_dense = LeakyReLU(alpha=0.2)(text_dense)
text_dense = Reshape(image_dim)(text_dense)
# Combine image and text
combined_input = Concatenate()([img_input, text_dense])
# Build the discriminator network
x = Conv2D(64, (4,4), strides=(2,2), padding='same')(combined_input)
x = LeakyReLU(alpha=0.2)(x)
x = Conv2D(128, (4,4), strides=(2,2), padding='same')(x)
x = LeakyReLU(alpha=0.2)(x)
x = Conv2D(256, (4,4), strides=(2,2), padding='same')(x)
x = LeakyReLU(alpha=0.2)(x)
x = Flatten()(x)
output = Dense(1, activation='sigmoid')(x)
model = Model([img_input, text_input], output)
return model
# Create discriminator
discriminator = build_discriminator(image_dim, vocab_size, embedding_dim, 
max_length)
discriminator.summary()
