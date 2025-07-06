pip install pandas scikit-learn tensorflow matplotlib
# UAS-KecerdasaanKomputasional
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# 1. Dataset minimal 20 data (10 hoaks, 10 asli)
data = pd.DataFrame({
    'text': [
        'Presiden resmikan jembatan baru di Kalimantan',
        'Minum kopi bisa menyembuhkan kanker',
        'BMKG prediksi cuaca ekstrem minggu depan',
        'Vaksin COVID-19 menyebabkan autisme',
        'Polisi tangkap pelaku pembobolan ATM',
        'Garam dan lemon bisa sembuhkan diabetes',
        'Menteri keuangan umumkan bantuan tunai',
        'Tidur miring ke kiri mempercepat kesembuhan penyakit jantung',
        'Jalan tol baru dibuka tanpa biaya',
        'Makan semangka tiap pagi bisa cegah COVID-19',
        'Petani panen raya di musim kemarau',
        'Makanan pedas bisa menyembuhkan tumor',
        'Bandara Soetta tambah penerbangan internasional',
        'Hidung gatal tanda akan menerima uang',
        'Universitas Indonesia buka prodi baru',
        'Minum air es setelah makan menyebabkan kanker',
        'Pemerintah beri subsidi untuk UMKM',
        'Mandi tengah malam bisa bikin lumpuh',
        'Banjir besar di Kalimantan berhasil diatasi',
        'Tidur tanpa bantal bikin otak lebih segar'
    ],
    'label': [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]  # 0 = asli, 1 = hoaks
})

# 2. Preprocessing teks
texts = data['text'].values
labels = data['label'].values

tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
max_length = 20
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, labels, test_size=0.2, random_state=42
)

# 4. CNN Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=1000, output_dim=64, input_length=max_length),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. Training
history = model.fit(
    X_train, y_train,
    epochs=15,
    validation_data=(X_test, y_test),
    verbose=2
)

# 6. Evaluasi
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))

# 7. Visualisasi Akurasi dan Loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Akurasi')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
