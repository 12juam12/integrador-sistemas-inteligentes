import os
import numpy as np
import tensorflow as tf

class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_dir, img_size, batch_size, subset, validation_split=0.2, shuffle=True):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.subset = subset
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.class_names = [
            "auto_alta_calidad_alto_precio",
            "auto_alta_calidad_bajo_precio",
            "auto_baja_calidad_alto_precio",
            "auto_baja_calidad_bajo_precio",
            "camioneta_alta_calidad_alto_precio",
            "camioneta_alta_calidad_bajo_precio",
            "camioneta_baja_calidad_alto_precio",
            "camioneta_baja_calidad_bajo_precio"
        ]

        self.image_paths, self.labels = self._load_data_paths_and_labels()
        self.on_epoch_end()

    def _load_data_paths_and_labels(self):
        image_paths = []
        labels = []
        for class_index, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.data_dir, *class_name.split('_'))
            print(f"Buscando en directorio: {class_dir}")
            for root, _, files in os.walk(class_dir):
                found_images = [f for f in files if f.lower().endswith(('png', 'jpg', 'jpeg'))]
                print(f"Encontradas {len(found_images)} imágenes JPEG en {root}")
                for file in found_images:
                    image_paths.append(os.path.join(root, file))
                    labels.append(class_index)

        data = list(zip(image_paths, labels))
        np.random.shuffle(data)

        if self.subset == 'training':
            data = data[:int(len(data) * (1 - self.validation_split))]
        else:
            data = data[int(len(data) * (1 - self.validation_split)):]

        image_paths, labels = zip(*data)
        print(f"Total de imágenes encontradas: {len(image_paths)}")
        return list(image_paths), list(labels)

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_image_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]

        images = []
        for img_path in batch_image_paths:
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(self.img_size, self.img_size))
            img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            images.append(img)

        return np.array(images), tf.keras.utils.to_categorical(batch_labels, num_classes=len(self.class_names))

    def on_epoch_end(self):
        if self.shuffle:
            data = list(zip(self.image_paths, self.labels))
            np.random.shuffle(data)
            self.image_paths, self.labels = zip(*data)

class MainTrainModel:
    def __init__(self, data_dir, img_size=150, batch_size=32):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.train_generator = None
        self.validation_generator = None
        self.model = None

    def cargar_datos(self):
        self.train_generator = CustomDataGenerator(
            data_dir=self.data_dir,
            img_size=self.img_size,
            batch_size=64,  
            subset='training'
        )
        self.validation_generator = CustomDataGenerator(
            data_dir=self.data_dir,
            img_size=self.img_size,
            batch_size=self.batch_size,
            subset='validation'
        )

    def definir_modelo(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_size, self.img_size, 3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(128, (5, 5), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(8, activation='softmax') 
        ])


        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

    def entrenar_modelo(self, epochs=20):
        self.model.fit(
            self.train_generator,
            epochs=epochs,
            batch_size=64,  
            validation_data=self.validation_generator,
            verbose=1
        )

    def guardar_modelo(self, filename='modelo_clasificacion.h5'):
        self.model.save(filename)
        print(f'Modelo guardado como {filename}')

    def cargar_y_preprocesar_imagen(self, ruta):
        img = tf.keras.preprocessing.image.load_img(ruta, target_size=(self.img_size, self.img_size))
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def clasificar_imagen(self, ruta_imagen):
        img = self.cargar_y_preprocesar_imagen(ruta_imagen)
        resultado = self.model.predict(img)
        print("Probabilidades de cada clase:", resultado)
        clase_predicha = np.argmax(resultado)
        print("Clase predicha:", clase_predicha)

        categorias = [
            "Auto de alta calidad y alto precio",
            "Auto de alta calidad y bajo precio",
            "Auto de baja calidad y alto precio",
            "Auto de baja calidad y bajo precio",
            "Camioneta de alta calidad y alto precio",
            "Camioneta de alta calidad y bajo precio",
            "Camioneta de baja calidad y alto precio",
            "Camioneta de baja calidad y bajo precio"
        ]

        return categorias[clase_predicha]

class MejoradoTresMainTrainModel(MainTrainModel):
    def __init__(self, data_dir, img_size=150, batch_size=32):
        super().__init__(data_dir, img_size, batch_size)