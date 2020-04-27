{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/kevin-home/.local/lib/python3.6/site-packages/keras/activations.py:235: UserWarning: Do not pass a layer instance (such as LeakyReLU) as the activation argument of another layer. Instead, advanced activation layers should be used just like any other layer in a model.\n",
      "  identifier=identifier.__class__.__name__))\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train on 60000 samples, validate on 10000 samples\n",
      "WARNING:tensorflow:Model failed to serialize as JSON. Ignoring... 'LeakyReLU' object has no attribute '__name__'\n",
      "Epoch 1/50\n",
      "first epoch\n",
      "34688/60000 [================>.............] - ETA: 2s - loss: 0.2206"
     ]
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-20-bdb5bd3ff804>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     93\u001b[0m                                         \u001b[0mshuffle\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     94\u001b[0m                                         \u001b[0mvalidation_data\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx_test\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mx_test\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 95\u001b[0;31m                                         callbacks=[saver,TensorBoard(log_dir=current_dir+'/tensorboard_data/test_i_g_a')])\n\u001b[0m\u001b[1;32m     96\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     97\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/.local/lib/python3.6/site-packages/keras/engine/training.py\u001b[0m in \u001b[0;36mfit\u001b[0;34m(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_freq, max_queue_size, workers, use_multiprocessing, **kwargs)\u001b[0m\n\u001b[1;32m   1237\u001b[0m                                         \u001b[0msteps_per_epoch\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0msteps_per_epoch\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1238\u001b[0m                                         \u001b[0mvalidation_steps\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mvalidation_steps\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1239\u001b[0;31m                                         validation_freq=validation_freq)\n\u001b[0m\u001b[1;32m   1240\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1241\u001b[0m     def evaluate(self,\n",
      "\u001b[0;32m~/.local/lib/python3.6/site-packages/keras/engine/training_arrays.py\u001b[0m in \u001b[0;36mfit_loop\u001b[0;34m(model, fit_function, fit_inputs, out_labels, batch_size, epochs, verbose, callbacks, val_function, val_inputs, shuffle, initial_epoch, steps_per_epoch, validation_steps, validation_freq)\u001b[0m\n\u001b[1;32m    194\u001b[0m                     \u001b[0mins_batch\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mi\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mins_batch\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mi\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtoarray\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    195\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 196\u001b[0;31m                 \u001b[0mouts\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mfit_function\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mins_batch\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    197\u001b[0m                 \u001b[0mouts\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mto_list\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mouts\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    198\u001b[0m                 \u001b[0;32mfor\u001b[0m \u001b[0ml\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mo\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mzip\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mout_labels\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mouts\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/.local/lib/python3.6/site-packages/tensorflow_core/python/keras/backend.py\u001b[0m in \u001b[0;36m__call__\u001b[0;34m(self, inputs)\u001b[0m\n\u001b[1;32m   3738\u001b[0m         \u001b[0mvalue\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mmath_ops\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcast\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mvalue\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtensor\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdtype\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3739\u001b[0m       \u001b[0mconverted_inputs\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mappend\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mvalue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 3740\u001b[0;31m     \u001b[0moutputs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_graph_fn\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0mconverted_inputs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   3741\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3742\u001b[0m     \u001b[0;31m# EagerTensor.numpy() will often make a copy to ensure memory safety.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/.local/lib/python3.6/site-packages/tensorflow_core/python/eager/function.py\u001b[0m in \u001b[0;36m__call__\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m   1079\u001b[0m       \u001b[0mTypeError\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mFor\u001b[0m \u001b[0minvalid\u001b[0m \u001b[0mpositional\u001b[0m\u001b[0;34m/\u001b[0m\u001b[0mkeyword\u001b[0m \u001b[0margument\u001b[0m \u001b[0mcombinations\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1080\u001b[0m     \"\"\"\n\u001b[0;32m-> 1081\u001b[0;31m     \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_call_impl\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1082\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1083\u001b[0m   \u001b[0;32mdef\u001b[0m \u001b[0m_call_impl\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mkwargs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcancellation_manager\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mNone\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/.local/lib/python3.6/site-packages/tensorflow_core/python/eager/function.py\u001b[0m in \u001b[0;36m_call_impl\u001b[0;34m(self, args, kwargs, cancellation_manager)\u001b[0m\n\u001b[1;32m   1119\u001b[0m       raise TypeError(\"Keyword arguments {} unknown. Expected {}.\".format(\n\u001b[1;32m   1120\u001b[0m           list(kwargs.keys()), list(self._arg_keywords)))\n\u001b[0;32m-> 1121\u001b[0;31m     \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_call_flat\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcaptured_inputs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcancellation_manager\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1122\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1123\u001b[0m   \u001b[0;32mdef\u001b[0m \u001b[0m_filtered_call\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/.local/lib/python3.6/site-packages/tensorflow_core/python/eager/function.py\u001b[0m in \u001b[0;36m_call_flat\u001b[0;34m(self, args, captured_inputs, cancellation_manager)\u001b[0m\n\u001b[1;32m   1222\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mexecuting_eagerly\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1223\u001b[0m       flat_outputs = forward_function.call(\n\u001b[0;32m-> 1224\u001b[0;31m           ctx, args, cancellation_manager=cancellation_manager)\n\u001b[0m\u001b[1;32m   1225\u001b[0m     \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1226\u001b[0m       \u001b[0mgradient_name\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_delayed_rewrite_functions\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mregister\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/.local/lib/python3.6/site-packages/tensorflow_core/python/eager/function.py\u001b[0m in \u001b[0;36mcall\u001b[0;34m(self, ctx, args, cancellation_manager)\u001b[0m\n\u001b[1;32m    509\u001b[0m               \u001b[0minputs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    510\u001b[0m               \u001b[0mattrs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"executor_type\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mexecutor_type\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m\"config_proto\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mconfig\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 511\u001b[0;31m               ctx=ctx)\n\u001b[0m\u001b[1;32m    512\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    513\u001b[0m           outputs = execute.execute_with_cancellation(\n",
      "\u001b[0;32m~/.local/lib/python3.6/site-packages/tensorflow_core/python/eager/execute.py\u001b[0m in \u001b[0;36mquick_execute\u001b[0;34m(op_name, num_outputs, inputs, attrs, ctx, name)\u001b[0m\n\u001b[1;32m     59\u001b[0m     tensors = pywrap_tensorflow.TFE_Py_Execute(ctx._handle, device_name,\n\u001b[1;32m     60\u001b[0m                                                \u001b[0mop_name\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0minputs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mattrs\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 61\u001b[0;31m                                                num_outputs)\n\u001b[0m\u001b[1;32m     62\u001b[0m   \u001b[0;32mexcept\u001b[0m \u001b[0mcore\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_NotOkStatusException\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     63\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mname\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D\n",
    "from keras.models import Model\n",
    "from keras import backend as K\n",
    "import keras\n",
    "import matplotlib.pyplot as plt\n",
    "import tensorflow as tf \n",
    "import cv2\n",
    "import os\n",
    "from keras.layers.advanced_activations import LeakyReLU\n",
    "\n",
    "\n",
    "\n",
    "input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format\n",
    "\n",
    "initial_mode = keras.initializers.lecun_normal(seed=None)\n",
    "initial_mode2 = keras.initializers.lecun_normal(seed=None)\n",
    "\n",
    "\n",
    "activation_method = LeakyReLU(alpha=0.2)\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "for i in range(1,2):\n",
    "    ## Build the Convolution autoencoders model\n",
    "    \n",
    "    x = Conv2D(16, (3, 3), activation=activation_method, padding='same',kernel_initializer=initial_mode)(input_img) # 28*28*16\n",
    "    x = MaxPooling2D((2, 2), padding='same')(x)  # 14*14*16\n",
    "    x = Conv2D(8, (3, 3), activation='relu', padding='same',kernel_initializer=initial_mode)(x) # 14*14*8\n",
    "    x = MaxPooling2D((2, 2), padding='same')(x) # 7*7*8\n",
    "    x = Conv2D(8, (3, 3), activation='relu', padding='same',kernel_initializer=initial_mode)(x) # 7*7*8\n",
    "    encoded = MaxPooling2D((2, 2), padding='same')(x) # 4*4*8\n",
    "\n",
    "    # This part is full connect layer\n",
    "    # at this point the representation is (4, 4, 8) i.e. 128-dimensional\n",
    "\n",
    "    x = Conv2D(8, (3, 3), activation='relu', padding='same',kernel_initializer=initial_mode)(encoded) # 4*4*8\n",
    "    x = UpSampling2D((2, 2))(x) # 8*8*8\n",
    "    x = Conv2D(8, (3, 3), activation='relu', padding='same',kernel_initializer=initial_mode)(x) # 8*8*8\n",
    "    x = UpSampling2D((2, 2))(x) # 16*16*8\n",
    "    x = Conv2D(16, (3, 3), activation='relu',kernel_initializer=initial_mode)(x) # 14*14*16\n",
    "    x = UpSampling2D((2, 2))(x) # 28*28*16\n",
    "    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same',kernel_initializer=initial_mode2)(x) # 28*28*1\n",
    "    \n",
    "\n",
    "\n",
    "    autoencoder = Model(input_img, decoded)\n",
    "    autoencoder.compile(optimizer='nadam', loss='binary_crossentropy')\n",
    "\n",
    "    from keras.datasets import mnist\n",
    "    import numpy as np\n",
    "\n",
    "    (x_train, _), (x_test, _) = mnist.load_data()\n",
    "\n",
    "    x_train = x_train.astype('float32') / 255.\n",
    "    x_test = x_test.astype('float32') / 255.\n",
    "    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format\n",
    "    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format\n",
    "\n",
    "    from keras.callbacks import TensorBoard\n",
    "\n",
    "    \n",
    "    \n",
    "\n",
    "    class CustomSaver(tf.keras.callbacks.Callback):\n",
    "        def on_epoch_begin(self, epoch, logs={}):            \n",
    "            loop = epoch + 1\n",
    "            \n",
    "            if epoch ==0:\n",
    "                print('first epoch')\n",
    "\n",
    "            \n",
    "            elif loop%5 == 0:  # or save after some epoch, each k-th epoch etc.\n",
    "                print('Save the image')\n",
    "                decoded_imgs = autoencoder.predict(x_test)\n",
    "                img = decoded_imgs[0]\n",
    "                img = img.reshape(28,28)\n",
    "                \n",
    "                # save the img\n",
    "                current_dir = os.getcwd()\n",
    "                plt.imshow(img,'gray')\n",
    "                plt.savefig(current_dir+'/Decoded_img/Decoded_image_epoch'+str(loop)+'.png')\n",
    "                \n",
    "                \n",
    "                \n",
    "\n",
    "    saver = CustomSaver()\n",
    "    current_dir = os.getcwd()\n",
    "\n",
    "    result = autoencoder.fit(x_train, x_train,\n",
    "                                        epochs=50,\n",
    "                                        batch_size=128,\n",
    "                                        shuffle=True,\n",
    "                                        validation_data=(x_test, x_test),\n",
    "                                        callbacks=[saver,TensorBoard(log_dir=current_dir+'/tensorboard_data/test_i_g_a')])\n",
    "    \n",
    "    \n",
    "\n",
    "\n",
    "#print(autoencoder.summary())\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "#decoded_images = autoencoder.predict(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "decoded_imgs = autoencoder.predict(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x7fadfe44f160>"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "b = x_test[0]\n",
    "b = b.reshape(28,28)\n",
    "\n",
    "plt.imshow(b,'gray')\n",
    "#plt.savefig('data_test_src.png')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x7fadfe3a2898>"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "decoded_imgs.shape\n",
    "a = decoded_imgs[0]\n",
    "a = a.reshape(28,28)\n",
    "plt.imshow(a,'gray')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "autoencoder.save('my_model.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D\n",
    "from tensorflow.keras.models import Model\n",
    "from tensorflow.keras import backend as K\n",
    "import matplotlib.pyplot as plt\n",
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "import cv2\n",
    "import os,sys\n",
    "\n",
    "current_dir = os.getcwd()\n",
    "new_model = tf.keras.models.load_model(current_dir+'/model/test_i_g.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"model_2\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "input_2 (InputLayer)         [(None, 28, 28, 1)]       0         \n",
      "_________________________________________________________________\n",
      "conv2d_8 (Conv2D)            (None, 28, 28, 16)        160       \n",
      "_________________________________________________________________\n",
      "max_pooling2d_4 (MaxPooling2 (None, 14, 14, 16)        0         \n",
      "_________________________________________________________________\n",
      "conv2d_9 (Conv2D)            (None, 14, 14, 8)         1160      \n",
      "_________________________________________________________________\n",
      "max_pooling2d_5 (MaxPooling2 (None, 7, 7, 8)           0         \n",
      "_________________________________________________________________\n",
      "conv2d_10 (Conv2D)           (None, 7, 7, 8)           584       \n",
      "_________________________________________________________________\n",
      "max_pooling2d_6 (MaxPooling2 (None, 4, 4, 8)           0         \n",
      "_________________________________________________________________\n",
      "conv2d_11 (Conv2D)           (None, 4, 4, 8)           584       \n",
      "_________________________________________________________________\n",
      "up_sampling2d_4 (UpSampling2 (None, 8, 8, 8)           0         \n",
      "_________________________________________________________________\n",
      "conv2d_12 (Conv2D)           (None, 8, 8, 8)           584       \n",
      "_________________________________________________________________\n",
      "up_sampling2d_5 (UpSampling2 (None, 16, 16, 8)         0         \n",
      "_________________________________________________________________\n",
      "conv2d_13 (Conv2D)           (None, 14, 14, 16)        1168      \n",
      "_________________________________________________________________\n",
      "up_sampling2d_6 (UpSampling2 (None, 28, 28, 16)        0         \n",
      "_________________________________________________________________\n",
      "conv2d_14 (Conv2D)           (None, 28, 28, 1)         145       \n",
      "=================================================================\n",
      "Total params: 4,385\n",
      "Trainable params: 4,385\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "print(new_model.summary())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(225, 225)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "current_dir = os.getcwd()\n",
    "\n",
    "test_img = cv2.imread('test1.png',cv2.IMREAD_GRAYSCALE)\n",
    "test_img.shape\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Do scale\n",
    "test_img = abs(255-test_img)\n",
    "test_img = test_img/255\n",
    "\n",
    "\n",
    "#change the input shape\n",
    "new_img = cv2.resize(test_img,(28, 28))\n",
    "new_img.shape\n",
    "plt.imshow(new_img,cmap='gray')\n",
    "#plt.savefig('test_src.png')\n",
    "\n",
    "new_img1 = new_img.reshape(1,28,28,1) \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1, 28, 28, 1)"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "new_img1.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "pred = new_model.predict(new_img1)\n",
    "\n",
    "output = pred.reshape(28,28) \n",
    "\n",
    "plt.imshow(output,cmap='gray')\n",
    "plt.savefig('test_zeros.png')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "tensor_env_3.6",
   "language": "python",
   "name": "tensor_env_3.6"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
