
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

# PSO parameters
N = 10  # Number of particles in the swarm
max_iter = 10  # Maximum number of iterations
c1 = 2.0  # Cognitive parameter
c2 = 2.0  # Social parameter
w = 0.7  # Inertia weight

# Dataset parameters
dataset_dir = "Alze"
class_labels = ["AD", "CN", "CI"]
image_shape = (128, 128) 

# Load and preprocess the image dataset
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(
    os.path.join(dataset_dir, "train"),
    target_size=image_shape,
    batch_size=16,  # Change batch size to 16
    class_mode="categorical"
)
test_generator = datagen.flow_from_directory(
    os.path.join(dataset_dir, "test"),
    target_size=image_shape,
    batch_size=16,  # Change batch size to 16
    class_mode="categorical",
    shuffle=False
)

num_classes = len(class_labels)
input_shape = (*image_shape, 3)


# Load and preprocess the image dataset
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(
    os.path.join(dataset_dir, "train"),
    target_size=image_shape,
    batch_size=32,
    class_mode="categorical"
)
test_generator = datagen.flow_from_directory(
    os.path.join(dataset_dir, "test"),
    target_size=image_shape,
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

num_classes = len(class_labels)
input_shape = (*image_shape, 3)

# CNN model
def create_model(selected_features):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Fitness function
def fitness_function(selected_features):
    model = create_model(selected_features)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    model.fit(train_generator, epochs=5, verbose=0)
    y_true = test_generator.classes
    y_pred = model.predict(test_generator).argmax(axis=1)
    accuracy = accuracy_score(y_true, y_pred)

    return accuracy

# Particle class
class Particle:
    def __init__(self, position):
        self.position = position
        self.velocity = np.zeros_like(position)
        self.best_position = position
        self.best_fitness = fitness_function(position)

# Particle Swarm Optimization
def particle_swarm_optimization():
    particles = []
    global_best_position = None
    global_best_fitness = 0.0

    num_features = train_generator[0][0][0].shape[-1]  # Number of features is determined by image shape

    # Initialize particles
    for _ in range(N):
        position = np.random.choice([0, 1], size=num_features)
        particle = Particle(position)
        particles.append(particle)

        if particle.best_fitness > global_best_fitness:
            global_best_fitness = particle.best_fitness
            global_best_position = particle.best_position

    # Main PSO loop
    for _ in range(max_iter):
        for particle in particles:
            # Update velocity
            new_velocity = (w * particle.velocity +
                            c1 * np.random.random() * (particle.best_position - particle.position) +
                            c2 * np.random.random() * (global_best_position - particle.position))
            new_position = np.where(new_velocity >= 0.5, 1, 0)
            particle.velocity = new_velocity
            particle.position = new_position

            fitness = fitness_function(particle.position)

            if fitness > particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position

            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle.position

    return global_best_position, global_best_fitness

# Main code
if __name__ == "__main__":
    best_position, best_fitness = particle_swarm_optimization()
    print("Best position:", best_position)
    print("Best fitness:", best_fitness)
