def train_gan(generator, discriminator, gan_model, epochs, batch_size):
for epoch in range(epochs):
for _ in range(batch_size):
# Generate random noise and text input
noise = np.random.normal(0, 1, (batch_size, noise_dim))
random_text = np.random.randint(0, vocab_size, (batch_size, 
max_length))
# Generate fake images
generated_images = generator.predict([random_text, noise])
# Get real images and text inputs 
real_images = np.random.randn(batch_size, 128, 128, 3) 
real_text = np.random.randint(0, vocab_size, (batch_size, 
max_length)) 
# Concatenate fake and real data for training discriminator
X_combined_images = np.concatenate([real_images, 
generated_images])
X_combined_text = np.concatenate([real_text, random_text])
y_combined = np.concatenate([np.ones((batch_size, 1)), 
np.zeros((batch_size, 1))])
discriminator.trainable = True
d_loss = discriminator.train_on_batch([X_combined_images, 
X_combined_text], y_combined)
noise = np.random.normal(0, 1, (batch_size, noise_dim))
random_text = np.random.randint(0, vocab_size, (batch_size, 
max_length))
y_gen = np.ones((batch_size, 1))
discriminator.trainable = False
g_loss = gan_model.train_on_batch([random_text, noise], y_gen)
print(f"Epoch {epoch}/{epochs}, Discriminator Loss: {d_loss}, 
Generator Loss: {g_loss}")
if epoch % 100 == 0:
noise = np.random.normal(0, 1, (1, noise_dim))
random_text = np.random.randint(0, vocab_size, (1, max_length))
generated_image = generator.predict([random_text, noise])
plt.imshow(generated_image[0])
plt.axis('off')
plt.show()
epochs = 10
train_gan(generator, discriminator, gan_model, epochs, batch_size)
