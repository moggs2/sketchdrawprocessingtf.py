import tensorflow as tf
import glob
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


filenames = glob.glob('sketchesdata/training.tfrecord-*')
raw_dataset = tf.data.TFRecordDataset(filenames)
raw_dataset
print(raw_dataset)
filenames = glob.glob('sketchesdata/eval.tfrecord-*')
valid_dataset = tf.data.TFRecordDataset(filenames)

for raw_record in raw_dataset.take(10):
      print(repr(raw_record))

for raw_record in raw_dataset.take(10):
      example = tf.train.Example()
      example.ParseFromString(raw_record.numpy())
      print(example)

feature_description = {
        "ink": tf.io.VarLenFeature(dtype=tf.float32),
        "shape": tf.io.FixedLenFeature([2], dtype=tf.int64),
        "class_index": tf.io.FixedLenFeature([1], dtype=tf.int64)
}

def parsing_data(datatotake):
        parsed_dataset=tf.io.parse_single_example(datatotake, feature_description)
        dense_sketches=tf.sparse.to_dense(parsed_dataset['ink'])
        dense_sketches=tf.reshape(dense_sketches, [-1,3])
        #zerotensor=tf.Variable(tf.zeros([400,3]))
        zeroarray=np.zeros([2000,3])
        zerotensor=tf.Variable(zeroarray, dtype=tf.float32)
        #dense_sketches=dense_sketches[:10][:]
        shape=parsed_dataset['shape']
        zerotensor[:shape[0],:].assign(dense_sketches)
        #zerotensor=tf.reshape(zerotensor,[1,2000,3])
        class_index=parsed_dataset['class_index']
        #class_index=tf.one_hot(class_index,345)
        #class_index=tf.reshape(class_index, [-1,1])
        return zerotensor, shape, class_index

parsed_dataset = raw_dataset.map(parsing_data, num_parallel_calls=8)
print(parsed_dataset)
parsed_dataset_valid = valid_dataset.map(parsing_data, num_parallel_calls=8)

y=0

for parsed_record in parsed_dataset.take(1):
      drawedlines=parsed_record[0]
      drawedlinesnp=drawedlines.numpy()
      drawedlinesnp=drawedlinesnp.reshape((-1,3))
      print(drawedlinesnp)
      lengtharray=drawedlinesnp.shape[0]
      print(lengtharray)
      i=0
      summedlines=np.zeros([lengtharray,2])
      while i < (lengtharray):
          if i == 0:
              summedlines[i][0]=drawedlinesnp[i][0]
              summedlines[i][1]=drawedlinesnp[i][1]
          else:
              summedlines[i][0]=last0+drawedlinesnp[i][0]
              summedlines[i][1]=last1+drawedlinesnp[i][1]
              if last2==0.:
                 plt.plot((last0, summedlines[i,0]),(last1, summedlines[i,1]), '.-')
              elif last2==1.:
                 plt.plot((last0,summedlines[i,0]),(last1,summedlines[i,1]), 'y:')
          last0=summedlines[i][0]
          last1=summedlines[i][1]
          last2=drawedlinesnp[i][2]
          i=i+1
      y=y+1
      print(summedlines)
      print('Class: ' + str(parsed_record[2]))
      plt.savefig('sketchdrawtest' + str(y)  + '.png')
      plt.close()

parsed_dataset=parsed_dataset.filter(lambda x1, x2, z1: z1[0]<=2)
parsed_dataset_valid=parsed_dataset_valid.filter(lambda x1, x2, z1: z1[0]<=2)
parsed_dataset=parsed_dataset.map(lambda x1, y1, z1: (x1[:100], tf.one_hot(z1,3)[0]))
parsed_dataset_valid=parsed_dataset_valid.map(lambda x1, y1, z1: (x1[:100], tf.one_hot(z1,3)[0]))

#parsed_dataset=parsed_dataset.take(50000)
parsed_dataset=parsed_dataset.shuffle(100)
#parsed_dataset=parsed_dataset.take(10000)

parsed_dataset=parsed_dataset.batch(30)
#parsed_dataset_valid=parsed_dataset_valid.take(10000)
parsed_dataset_valid=parsed_dataset_valid.batch(30)
parsed_dataset_valid.cache()
parsed_dataset_valid.prefetch(10000)


parsed_dataset.cache()
parsed_dataset.prefetch(10000)
print(parsed_dataset)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(100,3,)),
    tf.keras.layers.Conv1D(20,7,strides=2),
    tf.keras.layers.Conv1D(20,7,strides=1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(20,3,strides=2),
    tf.keras.layers.Conv1D(20,3,strides=1),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.LSTM(20, return_sequences=True),
    tf.keras.layers.LSTM(10, return_sequences=False),
    tf.keras.layers.Dense(3, activation="softmax")
    ])

model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.losses.CategoricalCrossentropy(from_logits=False),
  metrics=[tf.metrics.CategoricalAccuracy()]
  )


model.summary()

early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=10)
best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint("sequence_model.h5", save_best_only=True)
learningratecallbackchange=tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.0015 * 0.9 ** epoch)


fittingdiagram=model.fit(
  parsed_dataset,
  validation_data=parsed_dataset_valid,
  epochs=100,
  #callbacks=[best_checkpoint_callback, early_stopping_callback, learningratecallbackchange])
  #callbacks=[best_checkpoint_callback, early_stopping_callback]
  )
