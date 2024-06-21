import os
import numpy as np

class CustomDataGenerator:
    def __init__(self, data_dir, img_size=150, batch_size=32, subset='training', validation_split=0.2, shuffle=True):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.subset = subset
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.image_paths, self.labels = self._load_data_paths_and_labels()
        self.on_epoch_end()

    def _load_data_paths_and_labels(self):
        image_paths = []
        labels = []

        class_names = [
            "auto/alta_calidad/alto_precio",
            "auto/alta_calidad/bajo_precio",
            "auto/baja_calidad/alto_precio",
            "auto/baja_calidad/bajo_precio",
            "camioneta/alta_calidad/alto_precio",
            "camioneta/alta_calidad/bajo_precio",
            "camioneta/baja_calidad/alto_precio",
            "camioneta/baja_calidad/bajo_precio"
        ]

        found_images = False

        for class_name in class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            print(f"Buscando en directorio: {class_dir}")

            if not os.path.exists(class_dir):
                print(f"Directorio no encontrado: {class_dir}")
                continue

            files = [f for f in os.listdir(class_dir) if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')]
            if len(files) == 0:
                print(f"No se encontraron imágenes JPEG en {class_dir}")
                continue

            found_images = True
            for file in files:
                image_paths.append(os.path.join(class_dir, file))
                labels.append(class_names.index(class_name))

            print(f"Encontradas {len(files)} imágenes JPEG en {class_dir}")

        if not found_images:
            raise ValueError("No se encontraron imágenes JPEG en ninguna de las carpetas especificadas.")

        print(f"Total de imágenes JPEG encontradas: {len(image_paths)}")

        return image_paths, labels

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_image_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]

        images = []
        for img_path in batch_image_paths:
            # Aquí cargarías y preprocesarías las imágenes según sea necesario
            # Ejemplo básico para cargar y escalar la imagen:
            # img = load_img(img_path, target_size=(self.img_size, self.img_size))
            # img = img_to_array(img) / 255.0
            # images.append(img)
            pass

        return np.array(images), np.array(batch_labels)

    def on_epoch_end(self):
        if self.shuffle:
            data = list(zip(self.image_paths, self.labels))
            np.random.shuffle(data)
            self.image_paths, self.labels = zip(*data)

# Ejemplo de uso
if __name__ == "__main__":
    data_dir = './data'
    try:
        generator = CustomDataGenerator(data_dir)
        print(f"Data Generator creado con {len(generator)} batches.")
    except Exception as e:
        print(f"Error al crear el Data Generator: {e}")
