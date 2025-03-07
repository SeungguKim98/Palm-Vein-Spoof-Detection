import glob
import os
import tqdm
import easydict
import datetime
import numpy as np
import tensorflow as tf
from random import shuffle, seed
from keras_cv_attention_models import mobilevit

seed(1111)

FLAGS = easydict.EasyDict({"img_size": 256,
                           "img_ch": 3,
                           "pre_checkpoint": False,
                           "lr": 0.00001, # 0.000001,
                           "pre_checkpoint_path": "",
                           "train": False,
                           "epochs": 30,
                           "batch_size": 4,
                           "save_checkpoint": "checkpoint save path"})

optim = tf.keras.optimizers.Adam(FLAGS.lr)
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

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

def contrastive_loss(features1, features2, temperature=0.1):
    batch_loss = 0
    for i in range(len(features1)):
        normalized_features1 = tf.math.l2_normalize(features1[i], axis=-1)
        normalized_features2 = tf.math.l2_normalize(features2[i], axis=-1)

        similarity_matrix = tf.matmul(normalized_features1, normalized_features2, transpose_b=True)
        exp_similarity_matrix = tf.exp(similarity_matrix / temperature)
        exp_similarity_matrix = tf.clip_by_value(exp_similarity_matrix, 1e-10, 1e10)

        sum_exp_similarity_matrix = tf.reduce_sum(exp_similarity_matrix, axis=1, keepdims=True)
        sum_exp_similarity_matrix = tf.clip_by_value(sum_exp_similarity_matrix, 1e-10, 1e10)

        log_probabilities = tf.math.log(exp_similarity_matrix / sum_exp_similarity_matrix)

        loss = -tf.reduce_mean(tf.linalg.diag_part(log_probabilities))
        batch_loss += loss
    return batch_loss

# @tf.function
def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        predictions, features1, features2 = model(images, training=True)
        contrast_loss = 0.1*contrastive_loss(features1, features2)
        loss = loss_object(labels, predictions) + contrast_loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss.update_state(loss)
    train_accuracy.update_state(labels, predictions)

@tf.function
def test_step(model, images, labels):
    predictions, _1, _2 = model(images, training=False)
    test_accuracy.update_state(labels, predictions)

    labels_argmax = tf.cast(tf.argmax(labels, 1), tf.int32)
    predictions_argmax = tf.cast(tf.argmax(predictions, 1), tf.int32)

    test_fp(labels_argmax, predictions_argmax)
    test_fn(labels_argmax, predictions_argmax)
    test_tp(labels_argmax, predictions_argmax)
    test_tn(labels_argmax, predictions_argmax)
    return predictions, test_fp.result(), test_fn.result(), test_tp.result(), test_tn.result()

def main():
    model = mobilevit.MobileViT_FFC_ATTN_FFTSA_S(input_shape=(256, 256, 3), num_classes=2, activation="swish",classifier_activation="softmax", pretrained=None)
    model.summary()
    model.load_weights("weight path")


    if FLAGS.train:
        # train
        train_real_img = glob.glob('J:/VERA-1vs50-FID-3/dataset/VERA-1vs50-FID-3/2-fold/*/*/*.png')
        train_fake_img = glob.glob('J:/VERA-1vs50-FID-3/cyclegan/fake/2-fold-fake/*.png')



        train_real_lab = [1 for i in range(len(train_real_img))]
        train_fake_lab = [0 for i in range(len(train_fake_img))]
        train_img = train_real_img + train_fake_img
        train_lab = train_real_lab + train_fake_lab
        print("train_img 개수 : {}".format(len(train_img)))

        # tensorboard 셋팅
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = FLAGS.save_checkpoint + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)


        for epoch in range(FLAGS.epochs):
            train_loss.reset_states()
            train_accuracy.reset_states()


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

            print(
                f'Epoch {epoch + 1}, '
                f'train_loss: {train_loss.result()}, '
                f'train_accuracy: {train_accuracy.result() * 100}, '
            )

            with train_summary_writer.as_default():
                tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
                tf.summary.scalar('train_accuracy', train_accuracy.result(), step=epoch)

            # EPOCH 단위 checkpoint 생성
            model_dir = "{}/{}".format(FLAGS.save_checkpoint, epoch)
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)
                print("==============================================================================")
                print("Make {} path to save checkpoint files".format(model_dir))
                print("==============================================================================")
            ckpt_dir = model_dir + "/" + "modified_model_{}.h5".format(epoch)
            model.save_weights(ckpt_dir)

    else:
        print("========================")
        print("Test the images...!!!!!!")
        print("========================")


        test_real_img = glob.glob('test_real_img path')
        test_fake_img = glob.glob('test_fake_img path')

        test_real_lab = [1 for i in range(len(test_real_img))]
        test_fake_lab = [0 for i in range(len(test_fake_img))]
        test_img = test_real_img + test_fake_img
        test_lab = test_real_lab + test_fake_lab

        TE = list(zip(test_img, test_lab))
        test_img, test_lab = zip(*TE)
        test_img, test_lab = np.array(test_img), np.array(test_lab)
        test_gener = tf.data.Dataset.from_tensor_slices((test_img, test_lab))
        test_gener = test_gener.map(test_map)
        test_gener = test_gener.batch(1)
        test_gener = test_gener.prefetch(tf.data.experimental.AUTOTUNE)

        test_iter = iter(test_gener)
        test_idx = len(test_img)

        for i in range(test_idx):

            test_imgs, test_labs = next(test_iter)

            predict, val_fp, val_fn, val_tp, val_tn = test_step(model, test_imgs, test_labs)
            APCER = val_fn / (val_tp + val_fn)
            BPCER = val_fp / (val_fp + val_tn)
            ACER = (APCER + BPCER) / 2

        print("최종 정확도 = {} %".format(test_accuracy.result() * 100.))
        print("{}".format(APCER))
        print("{}".format(BPCER))
        print("{}".format(ACER))

if __name__ == "__main__":
    main()