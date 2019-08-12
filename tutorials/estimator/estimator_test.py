import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path


FLAGS = tf.app.flags

# flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
# flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
# flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
# flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS.DEFINE_string("model_dir", "/tmp/dssmrank/model", "model directory")
FLAGS.DEFINE_string("train_data", "/Users/zhen.huaz/work/git/dpsr-dssm-variant/data-ranker/train3", "training data source")
FLAGS.DEFINE_string("valid_data", "/Users/zhen.huaz/work/git/dpsr-dssm-variant/data-ranker/test3", "valid data source")
FLAGS.DEFINE_string("test_data", "/Users/zhen.huaz/work/git/dpsr-dssm-variant/data-ranker/", "test data source")
FLAGS.DEFINE_integer("clean_model_dir", 1, "remove model dir")
FLAGS.DEFINE_integer("num_epochs", 10, "num of epochs")
FLAGS.DEFINE_integer("train_batch_size", 64, "batch of size")
FLAGS.DEFINE_integer("eval_batch_size", 500, "batch of size")
FLAGS.DEFINE_integer("shuffle_batches", 10, "batch of shuffle")

class estimator_test:
    def __init__(self,model_dir,train_data,valid_data,shuffle_batches, num_epochs, train_batch_size,eval_batch_size,config):
        self.model_dir = model_dir
        self.train_data = train_data
        self.valid_data = valid_data
        self.shuffle_batches = shuffle_batches
        self.num_epochs = num_epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.config = config

    #  input pipeline
    def build_reader(self,input_files, shuffle_batches, num_epochs, batch_size):
        def _parser(record):
            keys_to_features = {
                'image_raw': tf.FixedLenFeature((), tf.string),
                'label': tf.FixedLenFeature((), tf.int64)
            }
            # 非batch的行驶
            parsed = tf.parse_single_example(record, keys_to_features)
            image = tf.decode_raw(parsed['image_raw'], tf.uint8)
            image = tf.cast(image, tf.float32)
            label = tf.cast(parsed['label'], tf.int32)
            return image, label

        def _input_fn():
            dataset = tf.data.TFRecordDataset(input_files)

            dataset = dataset.prefetch(batch_size)
            # num_epochs 为整个数据集的迭代次数
            dataset = dataset.repeat(num_epochs)

            if shuffle_batches > 0:
                dataset = dataset.shuffle()
            else:
                dataset = dataset.batch(batch_size, drop_remainder=False)

            if shuffle_batches>0:
                dataset = dataset.shuffle(buffer_size=batch_size * shuffle_batches)
                dataset = dataset.batch(batch_size, drop_remainder=False)

            dataset = dataset.map(_parser)
            iterator = dataset.make_one_shot_iterator()

            features, labels = iterator.get_next()
            return features, labels

        return _input_fn()

    # model pipeline
    def _model_fn(self,features, labels, mode):

        """Model function for cifar10 model"""
        # 输入层
        x = tf.reshape(features, [-1, 32, 32, 3])
        # 第一层卷积层
        x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[
            3, 3], padding='same', activation=tf.nn.relu, name='CONV1')
        x = tf.layers.batch_normalization(
            inputs=x, training=mode == tf.estimator.ModeKeys.TRAIN, name='BN1')
        # 第一层池化层
        x = tf.layers.max_pooling2d(inputs=x, pool_size=[
            3, 3], strides=2, padding='same', name='POOL1')

        # 你可以添加更多的卷积层和池化层 ……

        # 全连接层
        x = tf.reshape(x, [-1, 8 * 8 * 128])
        x = tf.layers.dense(inputs=x, units=512, activation=tf.nn.relu, name='DENSE1')

        # 你可以添加更多的全连接层 ……

        logits = tf.layers.dense(inputs=x, units=10, name='FINAL')

        # 预测
        predictions = {
            'classes': tf.argmax(input=logits, axis=1, name='classes'),
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
        }


        # 预测模式，即mode == tf.estimator.ModeKeys.PREDICT，必须提供的是predicitions。
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

            # 计算损失（对于 TRAIN 和 EVAL 模式）
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
        loss = tf.losses.softmax_cross_entropy(onehot_labels, logits, scope='LOSS')

        # 评估方法
        accuracy, update_op = tf.metrics.accuracy(
            labels=labels, predictions=predictions['classes'], name='accuracy')
        batch_acc = tf.reduce_mean(tf.cast(
            tf.equal(tf.cast(labels, tf.int64), predictions['classes']), tf.float32))
        tf.summary.scalar('batch_acc', batch_acc)
        tf.summary.scalar('streaming_acc', update_op)

        # 验证模式，即 mode == tf.estimator.ModeKeys.EVAL，必须提供的是 loss。
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss)

        # 训练模式，即mode == tf.estimator.ModeKeys.TRAIN，必须提供的是loss和train_op。
        if mode == tf.estimator.ModeKeys.TRAIN:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(
                    loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        eval_metric_ops = {
            'accuracy': (accuracy, update_op)
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    def train(self):
        self.estimator = tf.estimator.Estimator(_model_fn=self._model_fn,model_dir=FLAGS.model_dir)
        self.train_input_fn = self.build_reader(self.train_data,self.shuffle_batches,self.num_epochs,self.train_batch_size)
        self.eval_input_fn = self.build_reader(self.valid_data,0,1,self.eval_batch_size)

        # 方法1:
        # cifar10_classifier.train(input_fn=train_input_fn)
        # cifar10_classifier.evaluate(input_fn=eval_input_fn)

        # 方法2:
        self.train_spec = tf.estimator.TrainSpec(input_fn=self.train_input_fn,
                                                 max_steps=None if 'max_train_steps' not in self.params else
                                                 self.params['max_train_steps']
                                                 )
        self.valid_spec = tf.estimator.EvalSpec(input_fn=self.eval_input_fn,
                                                steps=None,
                                                start_delay_secs=300,  # start evaluating after N seconds
                                                throttle_secs=30,  # evaluate every N seconds
                                                )
        tf.estimator.train_and_evaluate(self.estimator, self.train_spec, self.valid_spec)

    def get_estimator(self,config):
        '''Return the model as a Tensorflow Estimator object.'''
        return tf.estimator.Estimator(model_fn=self._model_fn, config=config)

    def infer(self,argv=None):
        '''Run the inference and return the result.'''
        config = tf.estimator.RunConfig()
        config = config.replace(model_dir=FLAGS.saved_model_dir)
        estimator = self.get_estimator(config)
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=self.load_image(), shuffle=False)
        result = estimator.predict(input_fn=predict_input_fn)
        for r in result:
            print(r)

    def load_image(self):
        '''Load image into numpy array.'''
        images = np.zeros((10, 3072), dtype='float32')
        for i, file in enumerate(Path('predict-images/').glob('*.png')):
            image = np.array(Image.open(file)).reshape(3072)
            images[i, :] = image
        return images

def main(unused_argv):
    config = tf.estimator.RunConfig()
    config = config.replace(model_dir=FLAGS.model_dir)
    estimator_test = estimator_test(FLAGS.model_dir,FLAGS.train_data,FLAGS.valid_data,FLAGS.shuffle_batches, FLAGS.num_epochs, FLAGS.train_batch_size,FLAGS.eval_batch_size,config)
    estimator_test.train()

    tf.logging.info('Saving hyperparameters ...')
    # save_hp_to_json()



if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.main()