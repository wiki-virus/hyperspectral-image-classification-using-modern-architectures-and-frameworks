import tensorflow as tf
import numpy as np

with tf.device('/GPU:0'):
    X = np.array([-3, -2, -1, 0, 1, 2, 3, 5, 8, 12, 15], dtype=float)
    X_cnn = X.reshape(-1, 1, 1)
    Y = np.array([8, 4, 2, 2, 4, 8, 14, 32, 74, 158, 242], dtype=float)
    ys=Y/100

    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=16,kernel_size=1,input_shape=(1,1), activation='tanh'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=4,activation='tanh'),
        tf.keras.layers.Dense(units=2)
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=opt,
        loss='mean_squared_error'
    )

    his=model.fit(X_cnn, ys, epochs=2000)

    model.save("my_model.h5")
n=14
mod=tf.keras.models.load_model("my_model.h5")
prediction = mod.predict(np.array([[[n]]]))

print("Prediction:", prediction*100)
print(abs(((prediction*100)/((n*n)+n+2))*100 - 100),"% off ")