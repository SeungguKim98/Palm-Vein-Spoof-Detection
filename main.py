import glob
import os
import tqdm
import easydict
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from random import shuffle, seed
from keras_cv_attention_models import davit, maxvit, swin_transformer_v2, mobilevit, inceptionnext, gcvit

FLAGS = easydict.EasyDict({"img_size": 224,
                           "img_ch": 3,
                           "pre_checkpoint": False,
                           "lr": 0.000001, # 0.000001,
                           "pre_checkpoint_path": "",
                           "train": True,
                           "epochs": 30,
                           "batch_size": 4,
                           "save_checkpoint": "J:/PLUS-1vs50-FID/cyclegan/result/2-fold/cgvit_s/Adam_1e-6_avg5-8888/"})

seed(8888)
optim = tf.keras.optimizers.Adam(FLAGS.lr)
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
testing_loss = tf.keras.metrics.Mean(name='testing_loss')
testing_accuracy = tf.keras.metrics.CategoricalAccuracy(name='testing_accuracy')
postprocessing_test_loss = tf.keras.metrics.Mean(name='postprocessing_test_loss')
postprocessing_test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='postprocessing_test_accuracy')
original_train_loss = tf.keras.metrics.Mean(name='original_train_loss')
original_train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='original_train_accuracy')
original_valid_loss = tf.keras.metrics.Mean(name='original_valid_loss')
original_valid_accuracy = tf.keras.metrics.CategoricalAccuracy(name='original_valid_accuracy')

test_fp = tf.keras.metrics.FalsePositives(name='test_fp')
test_fn = tf.keras.metrics.FalseNegatives(name='test_fn')
test_tn = tf.keras.metrics.TrueNegatives(name='test_tn')
test_tp = tf.keras.metrics.TruePositives(name='test_tp')

def train_map(img_list, lab_list):
    img = tf.io.read_file(img_list)
    img = tf.image.decode_image(img, FLAGS.img_ch, expand_animations=False)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.image.per_image_standardization(img)
    lab = tf.cast(lab_list, tf.int32)
    lab = tf.one_hot(lab, 2)
    return img, lab

def test_map(img_list, lab_list):
    img = tf.io.read_file(img_list)
    img = tf.image.decode_image(img, FLAGS.img_ch, expand_animations=False)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.image.per_image_standardization(img)
    lab = tf.cast(lab_list, tf.int32)
    lab = tf.one_hot(lab, 2)
    return img, lab

@tf.function
def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss.update_state(loss)
    train_accuracy.update_state(labels, predictions)

@tf.function
def test_step(model, images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss.update_state(t_loss)
    test_accuracy.update_state(labels, predictions)

    labels_argmax = tf.cast(tf.argmax(labels, 1), tf.int32)
    predictions_argmax = tf.cast(tf.argmax(predictions, 1), tf.int32)

    test_fp(labels_argmax, predictions_argmax)
    test_fn(labels_argmax, predictions_argmax)
    test_tp(labels_argmax, predictions_argmax)
    test_tn(labels_argmax, predictions_argmax)
    return predictions, test_fp.result(), test_fn.result(), test_tp.result(), test_tn.result()

@tf.function
def testing_step(model, images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    testing_loss.update_state(t_loss)
    testing_accuracy.update_state(labels, predictions)
    return predictions

@tf.function
def postprocessing_test_step(model, images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    postprocessing_test_loss.update_state(t_loss)
    postprocessing_test_accuracy.update_state(labels, predictions)
    return predictions

@tf.function
def original_train_step(model, images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    original_train_loss.update_state(t_loss)
    original_train_accuracy.update_state(labels, predictions)
    return predictions

@tf.function
def original_valid_step(model, images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    original_valid_loss.update_state(t_loss)
    original_valid_accuracy.update_state(labels, predictions)
    return predictions

def main():

    # model = davit.DaViT_S(input_shape=(224, 224, 3), num_classes=2, classifier_activation="softmax", pretrained="imagenet")
    # model = maxvit.MaxViT_Small(input_shape=(224, 224, 3), num_classes=2, drop_connect_rate=0, classifier_activation="softmax", pretrained="imagenet")
    # model = swin_transformer_v2.SwinTransformerV2Small_window8(input_shape=(256, 256, 3), num_classes=2, classifier_activation="softmax", pretrained="imagenet")
    # model = mobilevit.MobileViT_S(input_shape=(256, 256, 3), num_classes=2, activation="swish", classifier_activation="softmax", pretrained=None)
    # model = inceptionnext.InceptionNeXtSmall(input_shape=(224, 224, 3), num_classes=2, activation="gelu", classifier_activation="softmax", pretrained="imagenet")
    model = gcvit.GCViT_Small(input_shape=(224, 224, 3), num_classes=2, activation="gelu", classifier_activation="softmax", pretrained="imagenet")
    model.summary()

    if FLAGS.train:
        # train
        train_real_img = glob.glob('J:/PLUS-1vs50-FID/dataset/PLUS-1vs50-FID/2-fold/*/*/*.png')
        train_fake_img = glob.glob('J:/PLUS-1vs50-FID/cyclegan/fake/2-fold-fake/*.png')
        # testing
        testing_real_img = glob.glob('J:/PLUS-1vs50-FID/dataset/PLUS-1vs50-FID/1-fold/*/*/*.png')
        testing_fake_img = glob.glob('J:/PLUS-1vs50-FID/cyclegan/fake/1-fold-fake/*.png')
        # postprocessing_test
        postprocessing_test_real_img = glob.glob('J:/PLUS-1vs50-FID/dataset/PLUS-1vs50-FID/1-fold/*/*/*.png')
        postprocessing_test_fake_img = glob.glob('J:/PLUS-1vs50-FID/cyclegan/fake/1-fold-fake-avg5/*.png')
        # original_train
        original_train_real_img = glob.glob('J:/PLUS-1vs50-FID/dataset/PLUS-1vs50-FID/2-fold/*/*/*.png')
        original_train_fake_img = glob.glob('J:/PLUS-1vs50-FID/cyclegan/fake/2-fold-fake/*.png')
        # original_valid
        # original_valid_real_img = glob.glob('J:/VERA-1vs50-FID/dataset/PLUS-1vs50-FID/1-fold/*/*/*.png')
        # original_valid_fake_img = glob.glob('J:/VERA-1vs50-FID/cut/fake/1-fold-fake-avg5/*.png')


        train_real_lab = [1 for i in range(len(train_real_img))]
        train_fake_lab = [0 for i in range(len(train_fake_img))]
        train_img = train_real_img + train_fake_img
        train_lab = train_real_lab + train_fake_lab
        print("train_img 개수 : {}".format(len(train_img)))

        testing_real_lab = [1 for i in range(len(testing_real_img))]
        testing_fake_lab = [0 for i in range(len(testing_fake_img))]
        testing_img = testing_real_img + testing_fake_img
        testing_lab = testing_real_lab + testing_fake_lab
        print("testing_img 개수 : {}".format(len(testing_img)))

        postprocessing_test_real_lab = [1 for i in range(len(postprocessing_test_real_img))]
        postprocessing_test_fake_lab = [0 for i in range(len(postprocessing_test_fake_img))]
        postprocessing_test_img = postprocessing_test_real_img + postprocessing_test_fake_img
        postprocessing_test_lab = postprocessing_test_real_lab + postprocessing_test_fake_lab
        print("postprocessing_test_img 개수 : {}".format(len(postprocessing_test_img)))

        original_train_real_lab = [1 for i in range(len(original_train_real_img))]
        original_train_fake_lab = [0 for i in range(len(original_train_fake_img))]
        original_train_img = original_train_real_img + original_train_fake_img
        original_train_lab = original_train_real_lab + original_train_fake_lab
        print("original_train_img 개수 : {}".format(len(original_train_img)))

        # original_valid_real_lab = [1 for i in range(len(original_valid_real_img))]
        # original_valid_fake_lab = [0 for i in range(len(original_valid_fake_img))]
        # original_valid_img = original_valid_real_img + original_valid_fake_img
        # original_valid_lab = original_valid_real_lab + original_valid_fake_lab

        # tensorboard 셋팅
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = FLAGS.save_checkpoint + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        testing = list(zip(testing_img, testing_lab))
        shuffle(testing)
        testing_img, testing_lab = zip(*testing)
        testing_img, testing_lab = np.array(testing_img), np.array(testing_lab)
        testing_gener = tf.data.Dataset.from_tensor_slices((testing_img, testing_lab))
        testing_gener = testing_gener.map(test_map)
        testing_gener = testing_gener.batch(1)
        testing_gener = testing_gener.prefetch(tf.data.experimental.AUTOTUNE)

        postprocessing_test = list(zip(postprocessing_test_img, postprocessing_test_lab))
        shuffle(postprocessing_test)
        postprocessing_test_img, postprocessing_test_lab = zip(*postprocessing_test)
        postprocessing_test_img, postprocessing_test_lab = np.array(postprocessing_test_img), np.array(postprocessing_test_lab)
        postprocessing_test_gener = tf.data.Dataset.from_tensor_slices((postprocessing_test_img, postprocessing_test_lab))
        postprocessing_test_gener = postprocessing_test_gener.map(test_map)
        postprocessing_test_gener = postprocessing_test_gener.batch(1)
        postprocessing_test_gener = postprocessing_test_gener.prefetch(tf.data.experimental.AUTOTUNE)

        original_train = list(zip(original_train_img, original_train_lab))
        shuffle(original_train)
        original_train_img, original_train_lab = zip(*original_train)
        original_train_img, original_train_lab = np.array(original_train_img), np.array(original_train_lab)
        original_train_gener = tf.data.Dataset.from_tensor_slices((original_train_img, original_train_lab))
        original_train_gener = original_train_gener.map(test_map)
        original_train_gener = original_train_gener.batch(1)
        original_train_gener = original_train_gener.prefetch(tf.data.experimental.AUTOTUNE)

        # original_valid = list(zip(original_valid_img, original_valid_lab))
        # shuffle(original_valid)
        # original_valid_img, original_valid_lab = zip(*original_valid)
        # original_valid_img, original_valid_lab = np.array(original_valid_img), np.array(original_valid_lab)
        # original_valid_gener = tf.data.Dataset.from_tensor_slices((original_valid_img, original_valid_lab))
        # original_valid_gener = original_valid_gener.map(test_map)
        # original_valid_gener = original_valid_gener.batch(1)
        # original_valid_gener = original_valid_gener.prefetch(tf.data.experimental.AUTOTUNE)

        for epoch in range(FLAGS.epochs):
            train_loss.reset_states()
            train_accuracy.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()
            testing_loss.reset_states()
            testing_accuracy.reset_states()
            postprocessing_test_loss.reset_states()
            postprocessing_test_accuracy.reset_states()
            original_train_loss.reset_states()
            original_train_accuracy.reset_states()
            # original_valid_loss.reset_states()
            # original_valid_accuracy.reset_states()

            train = list(zip(train_img, train_lab))
            shuffle(train)
            train_img, train_lab = zip(*train)
            train_img, train_lab = np.array(train_img), np.array(train_lab)
            train_gener = tf.data.Dataset.from_tensor_slices((train_img, train_lab))
            train_gener = train_gener.map(train_map)
            train_gener = train_gener.batch(FLAGS.batch_size)
            train_gener = train_gener.prefetch(tf.data.experimental.AUTOTUNE)

            train_iter = iter(train_gener)
            train_idx = len(train_img) // FLAGS.batch_size
            for i in tqdm.tqdm(range(train_idx), desc='Epoch Loop'):
                batch_images, batch_labels = next(train_iter)
                train_step(model, batch_images, batch_labels)

            # test_iter = iter(test_gener)
            # test_idx = len(test_img)
            # for j in range(test_idx):
            #     test_imgs, test_labs = next(test_iter)
            #     test_step(model, test_imgs, test_labs)

            testing_iter = iter(testing_gener)
            testing_idx = len(testing_img)
            for k in range(testing_idx):
                testing_imgs, testing_labs = next(testing_iter)
                testing_step(model, testing_imgs, testing_labs)

            postprocessing_test_iter = iter(postprocessing_test_gener)
            postprocessing_test_idx = len(postprocessing_test_img)
            for l in range(postprocessing_test_idx):
                postprocessing_test_imgs, postprocessing_test_labs = next(postprocessing_test_iter)
                postprocessing_test_step(model, postprocessing_test_imgs, postprocessing_test_labs)

            original_train_iter = iter( original_train_gener)
            original_train_idx = len(original_train_img)
            for m in range( original_train_idx):
                original_train_imgs,  original_train_labs = next( original_train_iter)
                original_train_step(model,  original_train_imgs,  original_train_labs)

            # original_valid_iter = iter(original_valid_gener)
            # original_valid_idx = len(original_valid_img)
            # for n in range(original_valid_idx):
            #     original_valid_imgs, original_valid_labs = next(original_valid_iter)
            #     original_valid_step(model, original_valid_imgs, original_valid_labs)

            print(
                f'Epoch {epoch + 1}, '
                f'train_loss: {train_loss.result()}, '
                f'train_accuracy: {train_accuracy.result() * 100}, '
                # f'test_loss: {test_loss.result()}, '
                # f'test_accuracy: {test_accuracy.result() * 100}, '
                f'testing_loss: {testing_loss.result()}, '
                f'testing_accuracy: {testing_accuracy.result() * 100}, '
                f'postprocessing_test_loss: {postprocessing_test_loss.result()}, '
                f'postprocessing_test_accuracy: {postprocessing_test_accuracy.result() * 100}, '
                f'original_train_loss: {original_train_loss.result()}, '
                f'original_train_accuracy: {original_train_accuracy.result() * 100}, '
                # f'original_valid_loss: {original_valid_loss.result()}, '
                # f'original_valid_accuracy: {original_valid_accuracy.result() * 100}, '
            )

            with train_summary_writer.as_default():
                tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
                tf.summary.scalar('train_accuracy', train_accuracy.result(), step=epoch)
                # tf.summary.scalar('test_accuracy', test_accuracy.result(), step=epoch)
                tf.summary.scalar('testing_accuracy', testing_accuracy.result(), step=epoch)
                tf.summary.scalar('postprocessing_test_accuracy', postprocessing_test_accuracy.result(), step=epoch)
                tf.summary.scalar('original_train_accuracy', original_train_accuracy.result(), step=epoch)
                # tf.summary.scalar('original_valid_accuracy', original_valid_accuracy.result(), step=epoch)

            # EPOCH 단위 checkpoint 생성
            model_dir = "{}/{}".format(FLAGS.save_checkpoint, epoch)
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)
                print("==============================================================================")
                print("Make {} path to save checkpoint files".format(model_dir))
                print("==============================================================================")
            ckpt_dir = model_dir + "/" + "modified_model_{}.h5".format(epoch)
            model.save_weights(ckpt_dir)

if __name__ == "__main__":
    main()