images = df["image"]
text_data = df["caption"]
dataset_dir = '/content/flickr8k/Images'
num_images_to_display = 5
image_files = os.listdir(dataset_dir)[:num_images_to_display]
for image_file in image_files:
image_path = os.path.join(dataset_dir, image_file)
img = mpimg.imread(image_path)
plt.imshow(img)
plt.axis('off') 
plt.show()
def text_preprocessing(data):
X = data.apply(lambda x: x.lower())
data = data.apply(lambda x: x.replace("[^A-Za-z]",""))
data = data.apply(lambda x: x.replace("\s+"," "))
data = data.apply(lambda x: " ".join([word for word in x.split() if
len(word)>1]))
texts = "startseq "+ data +" endseq"
return texts
texts = text_preprocessing(text_data)
texts[:10]
path_images = images.tolist()
print(path_images)
path_images.remove('image')
print(path_images)
#Iterate through the list of filenames
image_directory = "/content/images_U"
image_paths = []
for filename in path_images:
full_image_path = os.path.join(image_directory, filename)
image_paths.append(full_image_path)
print("Image Path\n:",image_paths)
def read_image(path, img_size=128):
img = load_img(path, color_mode='rgb', target_size=(img_size, img_size))
img = img_to_array(img)
img = img / 255.0
return img
def preprocess_text(texts):
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, 
padding='post')
vocab_size = len(tokenizer.word_index) + 1
return padded_sequences, vocab_size, max_length, tokenizer
def preprocess_images(image_paths, img_size=128):
images = np.array([read_image(path, img_size) for path in image_paths])
return images
padded_sequences, vocab_size, max_length, tokenizer = preprocess_text(texts)
print(f"Padded sequences:\n{padded_sequences}")
print(f"Vocabulary size: {vocab_size}")
print(f"Max length: {max_length}")
