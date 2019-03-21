import tensorflow as tf
import numpy as np
import scipy
from os import path
from skimage import io
from skimage import transform
from glob import glob

class LatentSearch:    

    XENTROPY = 'XENTROPY'
    MSE = 'MSE'
    SSIM = 'ssim'
    
    @staticmethod
    def getTargetName(target_path):
        return path.basename(target_path).split('.')[0]
    
    def getTargetMoments(self, n):
         raise NotImplementedError()
    
    def __init__(self,
                 xtarget_paths,
                 zsize,
                 xshape,
                 optimizer,
                 lr,
                 distance_function,
                 moments_penalty=[]):
        
        self.xtarget_vals = [self.loadTargetImage(p, xshape[:2]) for p in xtarget_paths]
        self.xtarget_vals  = np.concatenate(self.xtarget_vals, 0)
        
        self.xtarget_name = [self.getTargetName(p) for p in xtarget_paths]
        
        self.n = self.xtarget_vals.shape[0]
        
        self.zsize = zsize
        self.xshape = xshape
       
        self.optimizer = optimizer
        self.lr = lr
        
        self.distance_function = distance_function
        self.moments_penalty = moments_penalty
        self.other_ops = []
        
    @staticmethod
    def loadTargetImage(target_path, size):
        raw = io.imread(target_path)
        raw = transform.resize(raw, size, mode='reflect')
        xtarget = np.expand_dims(raw, 0)
        return xtarget

    def init_z0(self, size):
        raise NotImplementedError()
            
    def addConstraintToZhat(self, zhat):
        return zhat
    
    
    def getDistanceFunction(self, xhat, xtarget):
        """ IMPORTANT: The error value is for image no for batch """
        
        if self.distance_function == self.XENTROPY:
            xtarget_flat = tf.layers.flatten(xtarget)
            xhat_flat = tf.layers.flatten(xhat)
            loss = tf.losses.softmax_cross_entropy(logits=xhat_flat, onehot_labels=tf.nn.softmax(xtarget_flat), reduction=tf.losses.Reduction.NONE)

        elif self.distance_function == self.MSE:
            loss = tf.reduce_mean( tf.losses.mean_squared_error(xtarget, xhat , reduction=tf.losses.Reduction.NONE), (1, 2, 3))
            
        elif self.distance_function == self.SSIM:
            print(xtarget, xhat)
            ssim = 1 - tf.image.ssim(xtarget, xhat, 1.0)
            loss = ssim
            
        return loss
             
        
    def __call__(self, generator_function, iters, logs_num, log_path=None):
        
        self.iters = iters
        
        # init zhat
        z0 = self.init_z0(self.zsize)
        zhat_variable = tf.Variable(z0, dtype=tf.float32)
                    
        # add constraints on zhat (if any)
        zhat = self.addConstraintToZhat(zhat_variable)
        
        # take generator output
        xhat = generator_function(zhat)
        
        # Load in memory and resize to generetor output size all the xtarget values
        xtarget = tf.constant(self.xtarget_vals, tf.float32)

        # Distance between produced and target image (separate for each image)
        target_distance = self.getDistanceFunction(xhat, xtarget)

        # Moments penalty
        moments_penalty = self.loss_penalties(zhat)
        
        # Final loss function
        loss = target_distance + moments_penalty
    
        xhat_group = tf.contrib.gan.eval.image_reshaper(xhat)
        xtarget_group = tf.contrib.gan.eval.image_reshaper(xtarget)
              
        # Optimization operation (only on zhat)
        search_op = self.optimizer(self.lr).minimize(loss, var_list=[zhat_variable])
        
        if log_path:
            log_fq = (iters // logs_num)
            tf.summary.scalar('Loss', tf.reduce_mean(loss), family='Loss')
            tf.summary.scalar('Target_distance', tf.reduce_mean(target_distance), family='Loss')
            tf.summary.scalar('Moments_penalty', tf.reduce_mean(moments_penalty), family='Loss')

            tf.summary.image('xhats', xhat_group)
            tf.summary.histogram('zhats', zhat)
            
            merged = tf.summary.merge_all()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            if log_path:
                train_writer = tf.summary.FileWriter(log_path, sess.graph)
            
            # training loop
            for i in range(iters):
                
                if self.other_ops:
                    sess.run(self.other_ops)
                
                if logs_num and i % log_fq == 0:
                    tf.logging.info('%d / %d' % (i, iters))
                    if log_path:
                        summary = sess.run(merged)
                        train_writer.add_summary(summary, i)
                else:
                    sess.run(search_op)
            
            self.xhat_val, self.zhat_val = sess.run( [xhat, zhat] )
            
            xhat_group_val, xtarget_group_val = sess.run([xhat_group, xtarget_group])
            
        return xhat_group_val, xtarget_group_val
    
    
    @staticmethod   
    def moment(z, zmean, i):
        ss = tf.reduce_mean( (z - zmean) ** i, 1, keepdims=True)
        return ss
        
    def loss_penalties(self, zhat):
        
        moments_penalty = self.moments_penalty
        nm = len(moments_penalty)

        assert nm >= 2

        loss = 0
        m_targets = self.getTargetMoments(nm)

        zhat_mean, zhat_var = tf.nn.moments(zhat, axes=[1], keep_dims=True)

        if moments_penalty[0]:
            tf.logging.info("m1 is applied")
            loss += moments_penalty[0] * ( (zhat_mean - m_targets[0]) ** 2)
            tf.summary.scalar('m1', tf.reduce_mean(zhat_mean), family="Moments")

        if moments_penalty[1]:
            tf.logging.info("m2 is applied")
            loss += moments_penalty[1] * ( (zhat_var - m_targets[1]) ** 2)
            tf.summary.scalar('m2', tf.reduce_mean(zhat_var), family="Moments")

        for i in range(2, nm):
            if moments_penalty[i]:
                tf.logging.info("m%d is applied" % (i+1))
                mi = self.moment(zhat, zhat_mean, i+1)
                tf.summary.scalar('m%s' % (i+1), tf.reduce_mean(mi), family="Moments")
                loss += moments_penalty[i] * ( (mi - m_targets[i]) ** 2)            

        return tf.reshape(loss, (-1,))

   
class NormalLatentSearch(LatentSearch):
        
    def getTargetMoments(self, n):
        nr = scipy.stats.norm(0, 1)
        mm = [nr.moment(i) for i in range(1, n+1)]
        return mm
    
    def init_z0(self, size):
        return np.random.normal(0., 1., (self.n, size)).astype(np.float32)
    
    
class UniformLatentSearch(LatentSearch):
    
    hard_clipping = True
    epsilon = 0.01
    uniform_edge = 1
    
    def getTargetMoments(self, n):
        un = scipy.stats.uniform(-1, 2)
        mm = [un.moment(i) for i in range(1, n+1)]
        return mm
    
    def init_z0(self, size):
        return np.random.uniform(-(self.uniform_edge - self.epsilon), (self.uniform_edge - self.epsilon), (self.n, size)).astype(np.float32)
    
    def addConstraintToZhat(self, zhat):
        if self.hard_clipping:
            tf.logging.info('Hard clipping is used!')
            return tf.clip_by_value(zhat, -(self.uniform_edge - self.epsilon), (self.uniform_edge - self.epsilon))
        else:
            return zhat
