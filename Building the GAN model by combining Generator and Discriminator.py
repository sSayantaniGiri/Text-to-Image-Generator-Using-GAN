#GAN
def build_gan(generator, discriminator):
discriminator.trainable = False
text_input = Input(shape=(max_length,))
noise_input = Input(shape=(noise_dim,))
generated_images = generator([text_input, noise_input])
gan_output = discriminator([generated_images, text_input])
gan_model = Model([text_input, noise_input], gan_output)
return gan_model
# Compile models
generator = build_generator(vocab_size, embedding_dim, noise_dim, max_length)
discriminator = build_discriminator(image_dim, vocab_size, embedding_dim, 
max_length)
gan_model = build_gan(generator, discriminator)
discriminator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), 
loss='binary_crossentropy', metrics=['accuracy'])
gan_model.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), 
loss='binary_crossentropy', metrics=['accuracy'])
gan_model.summary()
