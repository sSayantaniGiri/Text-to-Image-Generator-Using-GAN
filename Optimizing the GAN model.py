def build_discriminator(vocab_size, embedding_dim, max_length, image_shape):
image_input = Input(shape=image_shape, name="image_input")
text_input = Input(shape=(max_length,), name="text_input")
text_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, 
input_length=max_length)(text_input)
text_embedding = Flatten()(text_embedding)
combined_input = Concatenate()([Flatten()(image_input), text_embedding])
x = Dense(512)(combined_input)
x = LeakyReLU(alpha=0.2)(x)
x = Dense(256)(x)
x = LeakyReLU(alpha=0.2)(x)
output = Dense(1, activation='sigmoid')(x)
model = Model([image_input, text_input], output)
return model
def build_gan(generator, discriminator):
discriminator.trainable = False
text_input, noise_input = generator.input
generated_image = generator.output
gan_output = discriminator([generated_image, text_input])
model = Model([text_input, noise_input], gan_output)
return model
image_shape = (128, 128, 3)
discriminator = build_discriminator(vocab_size, embedding_dim, max_length, 
image_shape)
discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', 
metrics=['accuracy'])
generator = build_generator(vocab_size, embedding_dim, noise_dim, max_length)
gan_model = build_gan(generator, discriminator)
gan_model.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')
def train_gan(generator, discriminator, gan_model, epochs, batch_size):
for epoch in range(epochs):
for _ in range(batch_size):
43
noise = np.random.normal(0, 1, (batch_size, noise_dim))
random_text = np.random.randint(0, vocab_size, (batch_size, 
max_length))
generated_images = generator.predict([random_text, noise])
real_images = np.random.randn(batch_size, 128, 128, 3) 
real_text = np.random.randint(0, vocab_size, (batch_size, 
max_length)) 
X_combined_images = np.concatenate([real_images, 
generated_images])
X_combined_text = np.concatenate([real_text, random_text])
y_combined = np.concatenate([np.ones((batch_size, 1)) * 0.9, 
np.zeros((batch_size, 1))])
discriminator.trainable = True
d_loss, d_acc = discriminator.train_on_batch([X_combined_images, 
X_combined_text], y_combined)
noise = np.random.normal(0, 1, (batch_size, noise_dim))
random_text = np.random.randint(0, vocab_size, (batch_size, 
max_length))
y_gen = np.ones((batch_size, 1))
discriminator.trainable = False
g_loss = gan_model.train_on_batch([random_text, noise], y_gen)
print(f"Epoch {epoch}/{epochs}, Discriminator Loss: {d_loss}, 
Discriminator Accuracy: {d_acc}, Generator Loss: {g_loss}")
if epoch % 100 == 0:
noise = np.random.normal(0, 1, (1, noise_dim))
random_text = np.random.randint(0, vocab_size, (1, max_length))
generated_image = generator.predict([random_text, noise])
plt.imshow((generated_image[0] * 127.5 + 127.5).astype(np.uint8))
plt.axis('off')
plt.show()
epochs = 10
train_gan(generator, discriminator, gan_model, epochs, batch_size)
